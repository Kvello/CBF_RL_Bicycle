import torch
import matplotlib.pyplot as plt
from math import ceil

def get_projection(Q:torch.Tensor, x_hom:torch.Tensor):
    # Construct second order equation for lambda(lagrange multiplier)
    a = (
        x_hom[0]**2 *Q[1,1]*Q[2,2] + 
        x_hom[1]**2 *Q[0,0]*Q[2,2] + 
        x_hom[2]**2 *Q[0,0]*Q[1,1]
    )
    b = -(
        x_hom[0]**2*(Q[1,1] + Q[2,2]) + 
        x_hom[1]**2*(Q[0,0] + Q[2,2]) +
        x_hom[2]**2*(Q[0,0] + Q[1,1])
    )
    c = (x_hom**2).sum()
    d = b**2 - 4*a*c
    if d < 0.0:
        raise ValueError("No real roots")
    sqrt_d = torch.sqrt(d)
    if 4*a*c < 0.0:
        raise ValueError("Two positive roots")
    else:
        lambda_p = (-b + sqrt_d)/(2*a)
    
    P = torch.eye(3) - lambda_p*Q
    xp = torch.linalg.solve(P, x_hom)
    # normalize:
    # Test solution
    P_inv = torch.linalg.inv(P)
    diff = xp.T@Q@xp
    diff2 = a*lambda_p**2 + b*lambda_p + c
    partial1 = (
        (1-lambda_p*Q[0,0])**(-2)*Q[0,0]*x_hom[0]**2 +
        (1-lambda_p*Q[1,1])**(-2)*Q[1,1]*x_hom[1]**2 +
        (1-lambda_p*Q[2,2])**(-2)*Q[2,2]*x_hom[2]**2
    )
    partial2 = (
        (1-lambda_p*Q[0,0])*
        (1-lambda_p*Q[1,1])*
        (1-lambda_p*Q[2,2])
    )**(-2)
    partial3 = (
        Q[0,0]*x_hom[0]**2*(1-lambda_p*Q[1,1])**2*(1-lambda_p*Q[2,2])**2 +
        Q[1,1]*x_hom[1]**2*(1-lambda_p*Q[0,0])**2*(1-lambda_p*Q[2,2])**2 +
        Q[2,2]*x_hom[2]**2*(1-lambda_p*Q[0,0])**2*(1-lambda_p*Q[1,1])**2
    )
    if  diff > 0.0:
        print(diff)
        print(diff2*partial2)
        print(partial2)
        print(partial1)
        print(partial3*partial2)
        print(lambda_p)
        print(Q)
    return xp

def plot_projection(
    Q:torch.Tensor, x_hom:torch.Tensor, 
    x_p:torch.Tensor, ax=None
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Plot the conic
    t = torch.linspace(0,ceil(1/f) , 100)
    x1 = A*torch.sin(2*torch.pi*f*t)
    x2 = A*2*torch.pi*f*torch.cos(2*torch.pi*f*t)
    ax.plot(x1, x2, label='Conic', color='blue')
    
    # Plot the point
    ax.scatter(x_hom[0], x_hom[1], color='red', label='Point')
    
    # Plot the projection
    ax.scatter(x_p[0], x_p[1], color='green', label='Projection')
    
    # Set labels and title
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Projection of Point onto Conic')
    
    # Show legend
    ax.legend()
    plt.show()

A = 0.5
f = 0.15
x1_0 = torch.tensor(0.75)
x2_0 = torch.tensor(-0.75)
x0_hom = torch.tensor([x1_0, x2_0, 1.0])

# Conic representation of parametric curve
Q = torch.tensor([[1.0/A,0.0,0.0],
                  [0.0,1.0/(A*2*torch.pi*f),0.0],
                  [0.0,0.0,-1.0]])


xp = get_projection(Q,x0_hom)
plot_projection(Q, x0_hom, xp)
