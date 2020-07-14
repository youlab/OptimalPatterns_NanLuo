% Predicting colony patterns with various seeding configurations
% Generates results in Figure 6 and Figure S8

clear
% Load parameter set
load('Parameters_multiseeding.mat'); % select parameter file
config  = 3; % select seeding configuration (see below)
NutrientLevel = 2;  % select nutrient level: 1-low, 2-medium, 3-high
N0      = N0s(NutrientLevel);
Width   = OptimalWidth(NutrientLevel);
Density = OptimalDensity(NutrientLevel);

% ------------------------ Seeding configurations -------------------------
switch config
    case 1; x0 = 0; y0 = 0; % one dot
    case 2; x0 = 17/2 * [-1, 1]; y0 = [0, 0]; % two dots side by side
    case 3; x0 = 38/2 * [-1, 1]; y0 = [0, 0]; % two dots side by side
    case 4; x0 = 19 * [-1, 0, 1]; y0 = [0, 0, 0]; % three dots side by side
    case 5; x0 = 10 * [0, sqrt(3)/2, -sqrt(3)/2]; y0 = 10 * [1, -0.5, -0.5]; % triangular
    case 6; x0 = 20 * [0, sqrt(3)/2, -sqrt(3)/2]; y0 = 20 * [1, -0.5, -0.5]; % triangular
    case 7; x0 = 15 * [-1, 1, 1, -1]; y0 = 15 * [1, 1, -1, -1]; % square
    case 8; x0 = 19 * [0, 0.5, 1, 0.5, -0.5, -1, -0.5]; % core-ring
            y0 = 19 * [0, sqrt(3)/2, 0, -sqrt(3)/2, -sqrt(3)/2, 0, sqrt(3)/2];
    case 9; x0 = 19 * [0, sqrt(2)/2, 1, sqrt(2)/2, 0, -sqrt(2)/2, -1, -sqrt(2)/2]; % ring
            y0 = 19 * [1, sqrt(2)/2, 0, -sqrt(2)/2, -1, -sqrt(2)/2, 0, sqrt(2)/2];
    case 10;x0 = 19 * [0, 0.3827, sqrt(2)/2, 0.9239, 1, 0.9239, sqrt(2)/2, 0.3827, 0, -0.3827, -sqrt(2)/2, -0.9239, -1, -0.9239, -sqrt(2)/2, -0.3827]; % ring
            y0 = 19 * [1, 0.9239, sqrt(2)/2, 0.3827, 0, -0.3827, -sqrt(2)/2, -0.9239, -1, -0.9239, -sqrt(2)/2, -0.3827, 0, 0.3827, sqrt(2)/2, 0.9239];
    case 11;x0 = [0, 0, 0, 0, 0, 0, 0, 0]; y0 = 6 * [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5]; % line
    case 12;ld = load('DUKE.mat'); Pattern = ld.D;
    case 13;ld = load('DUKE.mat'); Pattern = ld.U;
    case 14;ld = load('DUKE.mat'); Pattern = ld.K;
    case 15;ld = load('DUKE.mat'); Pattern = ld.E;
end

if config >= 12 && config <= 15
    Pattern = flipud(Pattern);
    [row,col] = find(Pattern > 0);
    row = row - (size(Pattern, 1) + 1) / 2;
    col = col - (size(Pattern, 2) + 1) / 2;
    domainsize = 42;
    x0 = col' * L / domainsize;
    y0 = row' * L / domainsize;
end
% -------------------------------------------------------------------------

% Parameters
L      = 90;
totalt = 48;

dt = 0.02;
nt = totalt / dt;
nx = 1001; ny = nx;
dx = L / (nx - 1); dy = dx;
x  = linspace(-L/2, L/2, nx);
y  = linspace(-L/2, L/2, ny);
[xx, yy] = meshgrid(x, y);

noiseamp = 0 * pi;

% Initialization
P = zeros(nx, ny);      % Pattern
C = zeros(nx, ny);      % Cell density
N = zeros(nx, ny) + N0; 

r0 = 5;    % initial radius 
C0 = 1.6;

nseeding = length(x0);
rr = zeros(nx, ny, nseeding);
for isd = 1 : nseeding
    rr(:,:,isd) = sqrt((xx - x0(isd)).^ 2 + (yy - y0(isd)) .^ 2);
end
rr = min(rr, [], 3);
P(rr <= r0) = 1;
C(P == 1) = C0 / (sum(P(:)) * dx * dy); C_pre = C;

% calculate the actual length of boundary of each inoculum
nseeding = length(x0);
nseg = 50; seglength = 2 * pi * r0 / nseg;
theta = linspace(0, 2 * pi, nseg + 1)'; theta = theta(1 : nseg);
colonyarray = polyshape(); % boundary of each colony
for iseed = 1 : nseeding
    colony = polyshape(r0 * sin(theta) + x0(iseed), r0 * cos(theta) + y0(iseed));
    colonyarray(iseed) = colony;
end
colonyunion = union(colonyarray); % joined boundary of all colonies
boundarylengths = zeros(nseeding, 1);
for iseed = 1 : nseeding
    colonyboundary = intersect(colonyunion.Vertices, colonyarray(iseed).Vertices, 'rows');
    boundarylengths(iseed) = seglength * size(colonyboundary, 1);
end
% ------------------------------------------------------------------------

ntips0 = ceil(boundarylengths * Density); % initial branch number
theta = []; Tipx = []; Tipy = [];
for iseed = 1 : nseeding
Tipxi = ones(ntips0(iseed), 1) * x0(iseed);  Tipx = [Tipx; Tipxi]; % x coordinates of every tip
Tipyi = ones(ntips0(iseed), 1) * y0(iseed);  Tipy = [Tipy; Tipyi]; % y coordinates of every tip
thetai = linspace(pi/2, 2 * pi+pi/2, ntips0(iseed) + 1)'; 
thetai = thetai(1 : ntips0(iseed)) + rand * pi; % growth directions of every branch
theta = [theta; thetai];
end
ntips0 = sum(ntips0);

dE = zeros(ntips0, 1);
BranchDomain = cell(ntips0, 1); % the domain covered by each branch
for k = 1 : ntips0; BranchDomain{k} = C > 0; end

Biomass = sum(C(:)) * (dx * dy);
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
    ntips = length(Tipx);
    
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
        dl = gama * dE / Width;
        if i == 0; dl = 0.5; end

        % Bifurcation
        R = 1.5 / Density;  % a branch will bifurcate if there is no other branch tips within the radius of R
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
            TipxO = Tipx + dl .* sin(thetaO);
            TipyO = Tipy + dl .* cos(thetaO);
            NO = interp2(xx, yy, N, TipxO, TipyO);
            [~, ind] = max(NO, [], 2); % find the direction with maximum nutrient
            for k = 1 : ntips
                Tipx(k) = TipxO(k, ind(k));
                Tipy(k) = TipyO(k, ind(k));
                theta(k) = thetaO(k, ind(k));
            end
        end

        % Growth stops when approaching edges
        ind = sqrt(Tipx.^2 + Tipy.^2) > 0.8 * L/2;
        Tipx(ind) = Tipx_pre(ind);
        Tipy(ind) = Tipy_pre(ind);

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
        drawnow
    
    end
    
end

