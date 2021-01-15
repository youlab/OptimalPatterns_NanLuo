function Biomass = OptimalModel2D(Width, Density, N0, gama)

% Parameters
if nargin < 1
    Width   = 4; 
    Density = 0.2;
    N0      = 8;
    gama    = 10;
end

% Parameters
L      = 90;
totalt = 24;

dt = 0.02;
nt = totalt / dt;
nx = 1001; ny = nx;
dx = L / (nx - 1); dy = dx;
x  = linspace(-L/2, L/2, nx);
y  = linspace(-L/2, L/2, ny);
[xx, yy] = meshgrid(x, y);
rr = sqrt(xx .^ 2 + yy .^ 2);

bN = 160;
DN = 9;
aC = 0.5;
KN = 0.8;
Cm = 0.05;

noiseamp = 0.5 * pi;

% Initialization
P = zeros(nx, ny);      % Pattern
C = zeros(nx, ny);      % Cell density
N = zeros(nx, ny) + N0; 

r0   = 5;    % initial radius 
C0   = 1.6;  % initial cell density 
ntips0 = ceil(2 * pi * r0 * Density); % initial branch number
ntips0 = max(ntips0, 2);  % initial branch number cannot be less than 2
P(rr <= r0) = 1;
C(P == 1) = C0 / (sum(P(:)) * dx * dy);
Tipx = zeros(ntips0, 1);  % x coordinates of every tip
Tipy = zeros(ntips0, 1);  % y coordinates of every tip
Biomass = sum(C(:)) * (dx * dy);

theta = linspace(pi/2, 2 * pi+pi/2, ntips0 + 1)'; 
theta = theta(1 : ntips0); % growth directions of every branch
delta = linspace(-1, 1, 201) * pi;

[MatV1N,MatV2N,MatU1N,MatU2N] = Diffusion(dx,dy,nx,ny,dt,DN);

for i = 0 : nt
    
    % -------------------------------------
    % Nutrient distribution and cell growth
    
    fN = N ./ (N + KN) .* Cm ./ (C + Cm) .* C;
    dN = - bN * fN;
    N  = N + dN * dt; 
    NV = MatV1N \ (N * MatU1N); N = (MatV2N * NV) / MatU2N;
    
    dC = aC * fN;
    C  = C + dC * dt; 
    
    % -------------------------------------
    % Branch extension and bifurcation
    ntips = length(Tipx);
    
    if mod(i, 0.5/dt) == 0
    
        Biomass_pre = Biomass;
        Biomass = sum(C(:)) * (dx * dy);
        dl  = gama * (Biomass - Biomass_pre) / (Width * ntips);
        if i == 0; dl = 0.5; end

        % Bifurcation
        R = 1.5 / Density;  % a branch will bifurcate if there is no other branch tips within the radius of R
        TipxNew = Tipx; TipyNew = Tipy; thetaNew = theta;
        for k = 1 : ntips
            dist2othertips = sqrt((TipxNew - Tipx(k)) .^ 2 + (TipyNew - Tipy(k)) .^ 2);
            dist2othertips = sort(dist2othertips);
            if dist2othertips(2) > R
                TipxNew = [TipxNew; Tipx(k) + dl * sin(theta(k) + 0.5 * pi)]; % splitting the old tip to two new tips
                TipyNew = [TipyNew; Tipy(k) + dl * cos(theta(k) + 0.5 * pi)]; 
                TipxNew(k) = TipxNew(k) + dl * sin(theta(k) - 0.5 * pi);
                TipyNew(k) = TipyNew(k) + dl * cos(theta(k) - 0.5 * pi);
                thetaNew = [thetaNew; theta(k)];
            end
        end
        Tipx = TipxNew; Tipy = TipyNew; theta = thetaNew; 

        ntips = length(Tipx);
        % Determine branch extension directions
        Tipx_pre = Tipx; Tipy_pre = Tipy;
        if i == 0
            Tipx = Tipx + dl .* sin(theta);
            Tipy = Tipy + dl .* cos(theta);
        else
            thetaO = ones(ntips, 1) * delta;
            TipxO = Tipx + dl .* sin(thetaO);
            TipyO = Tipy + dl .* cos(thetaO);
            NO = interp2(xx, yy, N, TipxO, TipyO);
            [~, ind] = max(NO, [], 2); % find the direction with maximum nutrient
            for k = 1 : ntips
                Tipx(k) = TipxO(k, ind(k));
                Tipy(k) = TipyO(k, ind(k));
                theta(k) = thetaO(k, ind(k)) + noiseamp * rand;
            end
        end

        % Growth stops when approaching edges
        ind = sqrt(Tipx.^2 + Tipy.^2) > 0.85 * L/2;
        Tipx(ind) = Tipx_pre(ind);
        Tipy(ind) = Tipy_pre(ind);

        % Fill the width of the branches    
        for k = 1 : ntips
            d = sqrt((Tipx(k) - xx) .^ 2 + (Tipy(k) - yy) .^ 2);
            P(d <= Width/2) = 1;     
        end
        C(P == 1) = Biomass / (sum(P(:)) * dx * dy);  % Make cell density uniform

        clf; ind = 1 : 2 : nx;
        subplot 121
            pcolor(xx(ind, ind), yy(ind, ind), C(ind, ind)); shading interp; axis equal;
            axis([-L/2 L/2 -L/2 L/2]); colormap('gray'); hold on
            set(gca,'YTick',[], 'XTick',[])
            plot(Tipx, Tipy, '.', 'markersize', 5)
            title 'Cell density'
        subplot 122
            pcolor(xx(ind, ind), yy(ind, ind), N(ind, ind)); shading interp; axis equal;
            axis([-L/2 L/2 -L/2 L/2]); set(gca,'YTick',[], 'XTick',[]);  
            colormap('parula'); caxis([0 N0])
            title 'Nutrient'
        drawnow
    
    end
    
end

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
