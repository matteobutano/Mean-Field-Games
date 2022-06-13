import numpy as np
import time as tm



xi = 0.5
nu = 0.5
L = 3
N = 100
gam = 0
s = 2

R = 0.37
m_0 = 2.5
mu =1
V = 10**20
g = -(2*mu*xi**2)/m_0
sigma = np.sqrt(2*xi*nu)
lam = -g*m_0 + gam*sigma**2*mu*np.log(np.sqrt(m_0))


dx = (2*L)/(N-1)
x = np.linspace(-L,L,N)
y = np.linspace(-L,L,N)
X,Y = np.meshgrid(x,y)

back_shift = list([ (i+1)%N for i in range(N)])
forward_shift = list([ (i-1)%N for i in range(N)])

def sup(a):
    lim= 10**(-6)
    ret = a*(a>lim) +np.full((a.shape[0],a.shape[1]),1)*(a<=lim)
    return ret
          
def norm(u,v):
    return np.sqrt(u**2+v**2)

C = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if np.sqrt((X[i,j])**2+ Y[i,j]**2) < R:
            C[i,j] = V

def L2_error(p, pn):
    return np.sqrt(np.sum((p - pn)**2)/np.sum(pn**2))       

def pjacobi(s,m,p):
    p[:,0] = np.sqrt(m_0)
    p[0,:] = np.sqrt(m_0)
    p[-1,:] = np.sqrt(m_0)
    p[:,-1] = np.sqrt(m_0)
    l2_target = 1e-8
    l2norm = 1
    while l2norm > l2_target:
        pn = p.copy()
        A = 2*mu*sigma**4 - lam*dx**2 - (g*m[1:-1,1:-1] + C[1:-1,1:-1])*dx**2 + mu*gam*sigma**2*np.log(sup(pn[1:-1,1:-1]))*dx**2
        Q = pn[1:-1,2:] + pn[1:-1, :-2] + pn[2:, 1:-1] + pn[:-2, 1:-1]
        p[1:-1,1:-1] = (0.5*mu*Q*sigma**4-0.5*mu*(sigma**2)*dx*s*(pn[2:,1:-1] - pn[:-2, 1:-1]))/A
        l2norm = L2_error(p,pn)
    return p

def qjacobi(s,m,q):
    q[:,0] = np.sqrt(m_0)
    q[0,:] = np.sqrt(m_0)
    q[-1,:] = np.sqrt(m_0)
    q[:,-1] = np.sqrt(m_0)
    l2_target = 1e-8
    l2norm = 1
    while l2norm > l2_target:
        qn = q.copy()
        A = 2*mu*sigma**4 - lam*dx**2 - (g*m[1:-1,1:-1] + C[1:-1,1:-1] - mu*gam*sigma**2*np.log(sup(m[1:-1,1:-1])))*dx**2 - mu*gam*sigma**2 * np.log(sup(qn[1:-1,1:-1]))*dx**2
        Q = qn[1:-1,2:] + qn[1:-1, :-2] + qn[2:, 1:-1] + qn[:-2, 1:-1]
        q[1:-1,1:-1] = (0.5*mu*Q*sigma**4+0.5*mu*(sigma**2)*dx*s*(qn[2:,1:-1] - qn[:-2, 1:-1]))/A
        l2norm = L2_error(q,qn)
    return q

def pjacobi_per(s,m,p):
    p[0,:]=np.sqrt(m_0)
    p[-1,:] = np.sqrt(m_0)
    l2_target = 1e-7
    l2norm = 1
    while l2norm > l2_target:
        pn = p.copy()
        A = 2*mu*sigma**4 - lam*dx**2 - (g*m[1:-1,:] + C[1:-1,:])*dx**2 + mu*gam*sigma**2 * np.log(sup(pn[1:-1,:]))*dx**2
        Q = pn[2:,:] + pn[:-2,:] + pn[1:-1,forward_shift] + pn[1:-1, back_shift]
        p[1:-1,:] = (0.5*mu*Q*sigma**4-0.5*mu*(sigma**2)*dx*s*(pn[2:,:] - pn[:-2,:]))/A
        l2norm = L2_error(p,pn)
    return p

def qjacobi_per(s,m,q):
    q[0,:]=np.sqrt(m_0)
    q[-1,:] = np.sqrt(m_0)
    l2_target = 1e-7
    l2norm = 1
    while l2norm > l2_target:
        qn = q.copy()
        A = 2*mu*sigma**4 - lam*dx**2 - (g*m[1:-1,:] + C[1:-1,:]- mu*gam*sigma**2*np.log(sup(m[1:-1,:])))*dx**2 - mu*gam*sigma**2 *np.log(sup(qn[1:-1,:]))*dx**2
        Q = qn[2:,:] + qn[:-2,:] + qn[1:-1,forward_shift] + qn[1:-1, back_shift]
        q[1:-1,:] = (0.5*mu*Q*sigma**4+0.5*mu*(sigma**2)*dx*s*(qn[2:,:] - qn[:-2,:]))/A
        l2norm = L2_error(q,qn)
    return q

def vel(m,p,q):
    phi_grad_x = (p[1:-1,2:]-p[1:-1,:-2])/(2*dx)
    phi_grad_y = (p[2:,1:-1]-p[:-2,1:-1])/(2*dx)
    gamma_grad_x = (q[1:-1,2:]-q[1:-1,:-2])/(2*dx)
    gamma_grad_y = (q[2:,1:-1]-q[:-2,1:-1])/(2*dx)
    v_x = sigma**2*(q[1:-1,1:-1]*phi_grad_x-p[1:-1,1:-1]*gamma_grad_x)/(2*m[1:-1,1:-1])
    v_y = sigma**2*(q[1:-1,1:-1]*phi_grad_y-p[1:-1,1:-1]*gamma_grad_y)/(2*m[1:-1,1:-1])-s
    return np.array([v_x,v_y])

def simulation(m,alpha,s):
    p = np.full((N,N),np.sqrt(m_0))
    q = np.full((N,N),np.sqrt(m_0))
    l2_target = 1e-7
    l2norm = 1
    start = tm.time()
    while l2norm > l2_target:
        mn = m.copy()
        p = pjacobi_per(s,mn,p.copy())
        q = qjacobi_per(s,mn,q.copy())
        m = alpha*p*q + (1-alpha)*mn
        l2norm = L2_error(m,mn)
        print(l2norm)
    end =tm.time()
    dur = end-start
    print('Convergence in',int(dur//60),'m',int(dur%60), 's.' )
    m = np.flip(m,0)
    
    return m,p,q


sim = simulation(np.full((N,N),m_0),0.1,s)

print(g,sigma)
m,p,q = sim[0],sim[1],sim[2]
v = vel(m,p,q)
vx,vy = v[0],v[1]

np.savetxt(r'data\m_N='+str(N)+'_L='+str(L)+'_xi='+str(round(xi,2))+'_nu='+str(round(nu,2))+'_gamma='+str(gam)+'_s='+str(s)+'.txt',m)
np.savetxt(r'data\p_N='+str(N)+'_L='+str(L)+'_xi='+str(round(xi,2))+'_nu='+str(round(nu,2))+'_gamma='+str(gam)+'_s='+str(s)+'.txt',p)
np.savetxt(r'data\vx_N='+str(N)+'_L='+str(L)+'_xi='+str(round(xi,2))+'_nu='+str(round(nu,2))+'_gamma='+str(gam)+'_s='+str(s)+'.txt',vx)
np.savetxt(r'data\vy_N='+str(N)+'_L='+str(L)+'_xi='+str(round(xi,2))+'_nu='+str(round(nu,2))+'_gamma='+str(gam)+'_s='+str(s)+'.txt',vy)


