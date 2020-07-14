% Predicting colony patterns under heterogeneous conditions (N0 gradient)
% Generates results in Figure 5

clear
% Load parameter set
load('Parameters_gradient_Figure5.mat'); % select parameter file

% Parameters
L      = 90;
totalt = 17.6;

dt = 0.02;
nt = totalt / dt;
nx = 1001; ny = nx;
dx = L / (nx - 1); dy = dx;
x  = linspace(-L/2, L/2, nx);
y  = linspace(-L/2, L/2, ny);
[xx, yy] = meshgrid(x, y);
rr = sqrt(xx .^ 2 + yy .^ 2);

noiseamp = 0 * pi;

% Initialization

P = zeros(nx, ny);      % Pattern
C = zeros(nx, ny);      % Cell density
N = gr * xx / L + Nc;
% Mapping patterns to local initial nutrient
Wmat = interp1(mapping_N, mapping_optimW, N, 'linear', 'extrap');
Dmat = interp1(mapping_N, mapping_optimD, N, 'linear', 'extrap');

r0   = 5; % initial radius 
C0   = 1.6;
ntips0 = 8;
P(rr <= r0) = 1;
C(P == 1) = C0 / (sum(P(:)) * dx * dy);
Tipx = zeros(ntips0, 1);
Tipy = zeros(ntips0, 1);
dE = zeros(ntips0, 1); dE_total = 0;
BranchRegion = cell(ntips0, 1);
for k = 1 : ntips0; BranchRegion{k} = C > 0; end

dtheta = pi/ntips0;
theta = linspace(dtheta, 2 * pi + dtheta, ntips0 + 1)';
theta = theta(1 : ntips0);

delta = linspace(-0.5, 0.5, 101) * pi;
[~,ind] = sort(abs(delta));
delta = delta(ind);

[MatV1N,MatV2N,MatU1N,MatU2N] = Branching_diffusion(dx,dy,nx,ny,dt,DN);

Biomass = sum(C(:)) * (dx * dy);
dBiomass = 0;

for i = 0 : nt
    
    fN = N ./ (N + KN) .* Cm ./ (C + Cm) .* C;
    dN = - bN * fN;
    N  = N + dN * dt; 
    NV = MatV1N \ (N * MatU1N); N = (MatV2N * NV) / MatU2N;
    
    dC = aC * fN;
    C  = C + dC * dt; 
    
    dBiomass = dBiomass + dC * dt * dx * dy;
    
    if mod(i, 0.2/dt) == 0   
      
        Width = interp2(xx, yy, Wmat, Tipx, Tipy);
        Biomass = sum(C(:)) * (dx * dy);  
        BranchRegionSum = cat(3, BranchRegion{:});
        BranchRegionSum = sum(BranchRegionSum, 3);
        ntips = length(Tipx);
        for k = 1 : ntips
            branchfract = 1 ./ (BranchRegionSum .* BranchRegion{k}); 
            branchfract(isinf(branchfract)) = 0;
            dE(k) = sum(sum(dBiomass .* sparse(branchfract)));
        end

        dl = gama * dE ./ Width;
        if i == 0; dl = 2; end
        
        % Bifurcation
        Density = interp2(xx, yy, Dmat, Tipx, Tipy); R = 3/2 ./ Density; 
        TipxNew = Tipx; TipyNew = Tipy; thetaNew = theta; dlNew = dl;
        BranchRegionNew = BranchRegion;
        for k = 1 : ntips
            dist2othertips = sqrt((TipxNew - Tipx(k)) .^ 2 + (TipyNew - Tipy(k)) .^ 2);
            dist2othertips = sort(dist2othertips);
            if dist2othertips(2) > R(k)
                TipxNew = [TipxNew; Tipx(k) + dl(k) * sin(theta(k) + 0.5 * pi)];
                TipyNew = [TipyNew; Tipy(k) + dl(k) * cos(theta(k) + 0.5 * pi)]; 
                TipxNew(k) = TipxNew(k) + dl(k) * sin(theta(k) - 0.5 * pi);
                TipyNew(k) = TipyNew(k) + dl(k) * cos(theta(k) - 0.5 * pi);
                dlNew = [dlNew; dl(k) / 2];
                dlNew(k) = dl(k) / 2;
                thetaNew = [thetaNew; theta(k)];
                BranchRegionNew{end+1} = BranchRegion{k};
            end
        end
        Tipx = TipxNew; Tipy = TipyNew; theta = thetaNew; dl = dlNew;
        BranchRegion = BranchRegionNew;
        
        ntips = length(Tipx);
        % Modifying branch extension directions
        if i > 0
            thetaO = theta + delta;
            TipxO = Tipx + dl .* sin(thetaO);
            TipyO = Tipy + dl .* cos(thetaO);
            NO = interp2(xx, yy, N, TipxO, TipyO);
            [~, ind] = max(NO, [], 2);
            for k = 1 : ntips
                theta(k) = thetaO(k, ind(k));
            end
            theta = theta + (rand(length(theta),1) - 0.5) * noiseamp;
        end

        % Branch tips moving outward
        ntips = length(Tipx);
        Tipx_pre = Tipx; Tipy_pre = Tipy;
        Tipx = Tipx + dl .* sin(theta);
        Tipy = Tipy + dl .* cos(theta);
        % Stop growing when approaching edges
        if max(sqrt(Tipx.^2 + Tipy.^2)) > 0.9 * L/2; break; end

        % Fill the width of the branches    
        Width = interp2(xx, yy, Wmat, Tipx, Tipy);    
        for k = 1 : ntips
            d = sqrt((Tipx(k) - xx) .^ 2 + (Tipy(k) - yy) .^ 2);
            P(d <= Width(k)/2) = 1;
            BranchRegion{k} = BranchRegion{k} | (d <= Width(k)/2);
        end
        C(P == 1) = Biomass / (sum(P(:)) * dx * dy);
        dE = zeros(ntips, 1); dE_total = 0;
        dBiomass = 0;
        
        clf; ind = 1 : 2 : nx;
        subplot 121
            pcolor(xx(ind, ind), yy(ind, ind), C(ind, ind)); shading interp; axis equal;
            axis([-L/2 L/2 -L/2 L/2]); colorbar; hold on
            set(gca,'YTick',[], 'XTick',[])
            plot(Tipx, Tipy, '.', 'markersize', 5)
        subplot 122
            pcolor(xx(ind, ind), yy(ind, ind), N(ind, ind)); shading interp; axis equal;
            axis([-L/2 L/2 -L/2 L/2]); set(gca,'YTick',[], 'XTick',[]);  
            colormap('parula'); colorbar; 
        drawnow
                
    end
    
end
