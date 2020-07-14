% Predicting colony patterns under uniform conditions (N0 & gama)
% Generates results in Figure 4 and Figure S5

clear
% Load parameter set
load('Parameters_uniform_Figure4.mat'); % select parameter file
panel   = 1; % select panel number
N0      = N0V(panel);
gama    = gamaV(panel);
Width   = OptimalWidth(panel);
Density = OptimalDensity(panel);

% Parameters
L      = 90;
totalt = 100;

dt = 0.02;
nt = totalt / dt;
nx = 1001; ny = nx;
dx = L / (nx - 1); dy = dx;
x  = linspace(-L/2, L/2, nx);
y  = linspace(-L/2, L/2, ny);
[xx, yy] = meshgrid(x, y);
rr = sqrt(xx .^ 2 + yy .^ 2);

noiseamp = 0;

% Initialization
P = zeros(nx, ny);      % Pattern
C = zeros(nx, ny);      % Cell density
N = zeros(nx, ny) + N0; 

r0   = 5;    % initial radius 
C0   = 1.6;
ntips0 = ceil(2 * pi * r0 * Density); % initial branch number
ntips0 = max(ntips0, 2);  % initial branch number cannot be less than 2
P(rr <= r0) = 1;
C(P == 1) = C0 / (sum(P(:)) * dx * dy); C_pre = C;
Tipx = zeros(ntips0, 1);  % x coordinates of every tip
Tipy = zeros(ntips0, 1);  % y coordinates of every tip

dE = zeros(ntips0, 1);
BranchDomain = cell(ntips0, 1); % the domain covered by each branch
for k = 1 : ntips0; BranchDomain{k} = C > 0; end

theta = linspace(pi/3, 2 * pi+pi/3, ntips0 + 1)'; 
theta = theta(1 : ntips0); % growth directions of every branch
delta = linspace(-1, 1, 201) * pi;

[MatV1N,MatV2N,MatU1N,MatU2N] = Branching_diffusion(dx,dy,nx,ny,dt,DN);

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
    
    if mod(i, 0.2/dt) == 0
    
        dBiomass = (C - C_pre) * dx * dy; 
        % compute the amount of biomass accumulation in each branch
        BranchDomainSum = cat(3, BranchDomain{:});
        BranchDomainSum = sum(BranchDomainSum, 3);
        ntips = length(Tipx);
        for k = 1 : ntips
            branchfract = 1 ./ (BranchDomainSum .* BranchDomain{k}); 
            branchfract(isinf(branchfract)) = 0;
            dE(k) = sum(sum(dBiomass .* sparse(branchfract)));
        end
        
        % extension rate of each branch
        dl = gama * dE ./ Width;
        if i == 0; dl = 0.5; end

        % Bifurcation
        R = 3/2 / Density;  % a branch will bifurcate if there is no other branch tips within the radius of R
        TipxNew = Tipx; TipyNew = Tipy; thetaNew = theta; dlNew = dl;
        BranchDomainNew = BranchDomain;
        for k = 1 : ntips
            dist2othertips = sqrt((TipxNew - Tipx(k)) .^ 2 + (TipyNew - Tipy(k)) .^ 2);
            dist2othertips = sort(dist2othertips);
            if dist2othertips(2) > R
                TipxNew = [TipxNew; Tipx(k) + dl(k) * sin(theta(k) + 0.5 * pi)]; % splitting the old tip to two new tips
                TipyNew = [TipyNew; Tipy(k) + dl(k) * cos(theta(k) + 0.5 * pi)]; 
                TipxNew(k) = TipxNew(k) + dl(k) * sin(theta(k) - 0.5 * pi);
                TipyNew(k) = TipyNew(k) + dl(k) * cos(theta(k) - 0.5 * pi);
                dlNew = [dlNew; dl(k) / 2];
                dlNew(k) = dl(k) / 2;
                thetaNew = [thetaNew; theta(k)];
                BranchDomainNew{end+1} = BranchDomain{k};
            end
        end
        Tipx = TipxNew; Tipy = TipyNew; theta = thetaNew; dl = dlNew;
        BranchDomain = BranchDomainNew;

        ntips = length(Tipx);
        % Determine branch extension directions
        Tipx_pre = Tipx; Tipy_pre = Tipy;
        if i == 0
            Tipx = Tipx + dl .* sin(theta);
            Tipy = Tipy + dl .* cos(theta);
        else
            thetaO = ones(ntips, 1) * delta;
            TipxO = Tipx + 0.05 .* sin(thetaO);
            TipyO = Tipy + 0.05 .* cos(thetaO);
            NO = interp2(xx, yy, N, TipxO, TipyO);
            [~, ind] = max(NO, [], 2); % find the direction with maximum nutrient
            TipxO = Tipx + dl .* sin(thetaO);
            TipyO = Tipy + dl .* cos(thetaO);
            for k = 1 : ntips
                Tipx(k) = TipxO(k, ind(k));
                Tipy(k) = TipyO(k, ind(k));
                theta(k) = thetaO(k, ind(k)) + noiseamp * rand;
            end
        end

        % Fill the width of the branches    
        for k = 1 : ntips
            d = sqrt((Tipx(k) - xx) .^ 2 + (Tipy(k) - yy) .^ 2);
            P(d <= Width/2) = 1;
            BranchDomain{k} = BranchDomain{k} | (d <= Width/2);  
        end
        C(P == 1) = sum(C(:)) / sum(P(:)); % Make cell density uniform
        C_pre = C;
        
        clf; ind = 1 : 2 : nx;
        subplot 121
            pcolor(xx(ind, ind), yy(ind, ind), C(ind, ind)); shading interp; axis equal;
            axis([-L/2 L/2 -L/2 L/2]); colormap('gray'); hold on
            set(gca,'YTick',[], 'XTick',[])
            plot(Tipx, Tipy, '.', 'markersize', 5)
        subplot 122
            pcolor(xx(ind, ind), yy(ind, ind), N(ind, ind)); shading interp; axis equal;
            axis([-L/2 L/2 -L/2 L/2]); set(gca,'YTick',[], 'XTick',[]);  
            colormap('parula'); caxis([0 N0])
            title(num2str(i * dt))
        drawnow
    
    end
    
end
