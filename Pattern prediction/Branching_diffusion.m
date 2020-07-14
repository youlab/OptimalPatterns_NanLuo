function [V1,V2,U1,U2] = Branching_diffusion(dx,dy,nx,ny,dt,D)
    
rx = dt / dx^2;
ry = dt / dy^2;
Ix = speye(nx); ex = ones(nx, 1); 
Iy = speye(ny); ey = ones(ny, 1); 

Mx = spdiags([ex -2*ex ex], -1:1, nx, nx);
My = spdiags([ey -2*ey ey], -1:1, ny, ny);
Mx(1, 2) = 2; Mx(nx, nx - 1) = 2;
My(2, 1) = 2; My(ny - 1, ny) = 2;

V1 = Ix - rx / 2 * D * Mx; V2 = Ix + rx / 2 * D * Mx;
U2 = Iy - ry / 2 * D * My; U1 = Iy + ry / 2 * D * My;  