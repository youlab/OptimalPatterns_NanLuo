function Biomass = OptimalModel1D(Width, Density, N0, gama)

% Parameters
if nargin < 1
    Width   = 5; 
    Density = 0.11;
    N0      = 8;
    gama    = 7.5;
end

Lx     = 100; % mm
Ly     = 100; % mm
totalt = 24;  % h

dt = 0.02;
nt = totalt / dt;
nx = 1000;
ny = 1000;
dx = Lx / (nx - 1);

bN = 160;
DN = 6;
aC = 0.8;
KN = 0.8;
Cm = 0.05;

C0 = 0.5;  % initial cell density
L  = 1;    % initial branch length

% Because of the symmetry, only one branch is simulated
Ld = 1 / Density;  % Growth domain of a single branch
nx = Ld / dx + 1;
[XP, nx] = Pattern(Width, 1, dx, nx);
XP = XP * ones(1, ny); % Pattern matrix

dx = Ld / nx;
dy = Ly / ny;
x  = linspace(-Ld/2, Ld/2, nx);
y  = linspace(0, ny * dy, ny);
[xx, yy] = meshgrid(x, y);
xx = xx'; yy = yy';

dLk = gama / (sum(XP(:, 1)) * dx);
Ln = round(L / dy);
C0 = C0 / (sum(XP(:, 1)) * dx * Ln * dy) / (Density * Lx);
C = C0 * XP;
C(:, Ln + 1 : end) = 0;
N = zeros(nx, ny) + N0;

[MatV1N,MatV2N,MatU1N,MatU2N] = Diffusion(dx,dy,nx,ny,dt,DN);

for i = 1 : nt
    
    % ------------ Growth ------------
    fN = N ./ (N + KN) .* Cm ./ (C + Cm) .* C; 
    dN = - bN * fN;
    N  = N + dN * dt; 
    N(N < 0) = 0; 
    NV = MatV1N \ (N * MatU1N); N = (MatV2N * NV) / MatU2N;
    
    dC = aC * fN;
    if Ln >= ny; dC = 0 * dC; end
    C  = C + dC * dt; 
    Biomass = sum(C(:)) * (dx * dy);
    
    % ----------- Movement -------------
    dL  = dLk * sum(sum(fN)) * (dx * dy);
    if Ln >= ny; dL = 0; end
    L = min(Ly, L + dL * dt);
    Ln = round(L / dy);
        
    C = XP * (Biomass / ((sum(XP(:, 1)) * Ln * dy * dx)));
    if Ln <= ny; C(:, Ln + 1 : end) = 0; end
    
    % ------------- Plot ---------------
    if mod(i, 50) == 0
    disp(i * dt); 
    subplot(1, 2, 1)
        pcolor(yy, xx, C); shading interp; axis equal; 
        axis([0 Ly -Ld/2 Ld/2]); title 'Cell density'
    subplot(1, 2, 2)
        pcolor(yy, xx, N); shading interp; axis equal;
        axis([0 Ly -Ld/2 Ld/2]); title 'Nutrient'
    drawnow
    end
end

Biomass = Biomass * Density * Lx;

function [X, nx] = Pattern(Width, Number, dx, nx)

nw = round(Width / dx);
nd = round(nx / Number);

X  = zeros(nd, Number);
nx = nd * Number;

X(1 : nw, :) = 1; X = flipud(X);
X = X(:);
X = X(1 : nx);
n = fix((nd - nw)/2); 
if n > 0; X = [X(n + 1 : end); X(1 : n)]; end


function [V1,V2,U1,U2] = Diffusion(dx,dy,nx,ny,dt,D)
    
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

