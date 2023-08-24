import math
import numpy as np
import functools
from scipy.special import sph_harm
import torch

def sph_sample(n, mode='DH'):
    """ Sample grid on a sphere.

    Args:
        n (int): dimension is n x n
        mode (str): sampling mode; DH or GLQ

    Returns:
        theta, phi (1D arrays): polar and azimuthal angles
    """
    assert n % 2 == 0
    j = np.arange(0, n)
    if mode == 'DH':
        return j*np.pi/n, j*2*np.pi/n
    elif mode == 'ours':
        return (2*j+1)*np.pi/2/n, j*2*np.pi/n
    elif mode == 'GLQ':
        from pyshtools.shtools import GLQGridCoord
        phi, theta = GLQGridCoord(n-1)
        # convert latitude to [0, np.pi/2]
        return np.radians(phi+90), np.radians(theta)
    elif mode == 'naive':
        # repeat first and last points; useful for plotting
        return np.linspace(0, np.pi, n), np.linspace(0, 2*np.pi, n)


# cache outputs; 2050 > 32*64
@functools.lru_cache(maxsize=2050, typed=False)
def sph_harm_lm(l, m, n):
    """ Wrapper around scipy.special.sph_harm. Return spherical harmonic of degree l and order m. """
    phi, theta = sph_sample(n)
    phi, theta = np.meshgrid(phi, theta)
    f = sph_harm(m, l, theta, phi)
    return f

def is_real_sft(h_or_c):
    """ Detect if list of lists of harmonics or coefficients assumes real inputs (m>0) """
    d = len(h_or_c[1])
    isreal = True if d == 2 else False
    return isreal

def sph_harm_to_shtools(c):
    """ Convert our list format for the sph harm coefficients/harmonics to pyshtools (2, n, n) format. """
    n = len(c)
    real = is_real_sft(c)
    dim1 = 1 if real else 2
    out = np.zeros((dim1, n, n, *c[0][0].shape)) + 0j
    for l, cc in enumerate(c):
        cc = np.array(cc)
        if not real:
            m_minus = cc[:l][::-1]
            m_plus = cc[l:]
        else:
            m_minus = np.array([])
            m_plus = cc

        # we get warnings here when using reals
        if m_minus.size > 0:
            out[1, l, 1:l+1, ...] = m_minus
        out[0, l, :l+1, ...] = m_plus
    return out


def sph_harm_all(n, as_tfvar=False, real=False):
    """ Compute spherical harmonics for an n x n input (degree up to n // 2)

    Args:
        n (int): input dimensions; order will be n // 2
        as_tfvar (bool): if True, return as list of tensorflow Variables.
        real (bool): if True, return real harmonics
    """
    harmonics = []
    for l in range(n // 2):
        if real:
            minl = 0
        else:
            minl = -l
        row = []
        for m in range(minl, l+1):
            row.append(sph_harm_lm(l, m, n))
        harmonics.append(row)

    if as_tfvar:
        harmonics = sph_harm_to_shtools(harmonics)
        return torch.complex(torch.FloatTensor(np.real(harmonics)), torch.FloatTensor(np.imag(harmonics)))
    else:
        return harmonics


def DHaj(n, mode='DH'):
    """ Sampling weights. """
    # Driscoll and Healy sampling weights (on the phi dimension)
    # note: weights depend on the chosen grid, given by sph_sample
    if mode == 'DH':
        def gridfun(j): return np.pi*j/n
    elif mode == 'ours':
        def gridfun(j): return np.pi*(2*j+1)/2/n
    else:
        raise NotImplementedError()

    l = np.arange(0, n/2)
    a = [(2*np.sqrt(2)/n * np.sin(gridfun(j)) * (1/(2*l+1) * np.sin((2*l+1)*gridfun(j))).sum()) for j in range(n)]
    return torch.FloatTensor(a).reshape((1,1,n,1))


def sphconv_op(f,g, harmonics, aj):
    device = f.device

    spectral_input = True if len(f.size())==5 else False
    spectral_filter = True if len(g.size())==5 else False
    
    n = f.size(2)
    if spectral_input:
        n *= 2
    
    if not spectral_input:
        cf = sph_harm_transform_batch(f, harmonics, aj, m0_only=False)
    else:
        cf = f

    if not spectral_filter:
        cg = sph_harm_transform_batch(g, harmonics, aj, m0_only=True)
    else:
        cg = g

    assert cf.size(4) == cg.size(0)
    assert cf.size(2) == cg.size(2)

    # per degree factor
    factor = 2*math.pi*torch.sqrt(4*math.pi/(2*torch.arange(n/2)+1))
    factor = torch.FloatTensor(factor).to(device).reshape((1,1,n//2,1,1,1))
    factor = torch.complex(factor, factor)

    cf = cf.unsqueeze(5)                                   # b*2*n/2*n/2*c*1

    cg = cg.permute(1,2,3,0,4).contiguous().unsqueeze(0)   # 1*1*n/2*1*c*filters
    real_cg = cg
    imag_cg = torch.zeros(cg.size()).to(cg.device)
    cg = torch.complex(real_cg, imag_cg)
    cfg = torch.sum(factor * cf * cg, dim=4)

    # import ipdb; ipdb.set_trace()
    # cfg = ((factor * cf) @ cg).squeeze(4)

    return sph_harm_inverse_batch(cfg, harmonics)


def sph_harm_transform_batch(f, harmonics, aj, m0_only=False):   
    """ Spherical harmonics batch-transform.

    Args:
        f (b, n, n, c)-array : functions are on l x l grid
        m0_only (bool): return only coefficients with order 0;
                        only them are needed when computing convolutions

    Returns:
        coeffs ((b, 2, n/2, n/2, c)-array):

    Params:
        harmonics (2, n/2, n/2, n, n)-array:
    """
    assert f.size(1) == f.size(2)
    n = f.size(1)

    harmonics = harmonics.clone()  # 2*n/2*n/2*n*n
    assert harmonics.size(0) in [1,2]        
    assert harmonics.size(1) == n//2
    assert harmonics.size(2) == n//2
    assert harmonics.size(3) == n
    assert harmonics.size(4) == n

    aj = aj.clone() # n

    if m0_only:
        harmonics = harmonics[slice(0, 1), :, slice(0, 1), ...]

    real_f = f*aj
    imag_f = torch.zeros(f.size()).to(f.device)
    f = torch.complex(real_f, imag_f)

    coeffs = torch.tensordot(f, torch.conj(harmonics), [[1, 2], [3, 4]])
    coeffs = (2*np.sqrt(2)*math.pi/n * coeffs).permute(0,2,3,4,1).contiguous()

    return coeffs


def sph_harm_inverse_batch(f, harmonics):
    """ Spherical harmonics batch inverse transform.

    Args:
        f ((b, 2, n/2, n/2, c)-array): sph harm coefficients; max degree is n/2
        harmonics (2, n/2, n/2, n, n)-array:

    Returns:
        recons ((b, n, n, c)-array):

    """

    n = f.size(2) *2

    harmonics = harmonics.clone()  # 2*n/2*n/2*n*n
    assert harmonics.size(0) in [1,2]        
    assert harmonics.size(1) == n//2
    assert harmonics.size(2) == n//2
    assert harmonics.size(3) == n
    assert harmonics.size(4) == n

    real = True if harmonics.size(0) == 1 else False

    if real:
        # using m, -m symmetry:
        # c^{-m}Y^{-m} + c^mY^m = 2(Re(c^{m})Re(Y^m) - Im(c^{m})Im(Y^m))
        # that does not apply to c_0 so we compensate by dividing it by two
        factor = torch.ones(f.size()[1:]).unsqueeze(0).to(f.device)
        factor[..., 0, :] = factor[..., 0, :]/2
        factor = torch.complex(factor, factor)

        recons = torch.tensordot(torch.real(f*factor), torch.real(harmonics), [[1, 2, 3], [0, 1, 2]]) - \
            torch.tensordot(torch.imag(f), torch.imag(harmonics),  [[1, 2, 3], [0, 1, 2]])
        recons = (2*recons).permute(0,2,3,1).contiguous()

    else:
        recons = torch.tensordot(f, harmonics, [[1, 2, 3], [0, 1, 2]])
        recons = recons.permute(0,2,3,1).contiguous()

    return recons