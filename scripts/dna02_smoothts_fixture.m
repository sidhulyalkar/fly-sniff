function dna02_smoothts_fixture(output_path)
%DNA02_SMOOTHTS_FIXTURE Export synthetic smoothts numerical fixtures.
%
% This function touches no fly data. It probes the installed MATLAB
% Financial Toolbox implementation of smoothts on fixed synthetic vectors
% so Python reproduction code can be qualified against actual MATLAB
% behavior before Figure 3C behavior is opened.

if nargin < 1 || isempty(output_path)
    output_path = fullfile('data', 'cache', 'dna02-smoothts-fixture-v1', ...
        'smoothts-fixture-v1.json');
end

if exist('smoothts', 'file') == 0
    error('fly_sniff:smoothts_missing', ...
        ['smoothts is not available on the MATLAB path. ', ...
         'Install/use a MATLAB release with Financial Toolbox smoothts support.']);
end

if exist('jsonencode', 'builtin') == 0 && exist('jsonencode', 'file') == 0
    error('fly_sniff:jsonencode_missing', ...
        'This fixture requires a MATLAB release that provides jsonencode.');
end

out_dir = fileparts(output_path);
if ~isempty(out_dir) && ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

fixture = struct();
fixture.schema = 'fly-sniff-dna02-smoothts-fixture-v1';
fixture.status = 'MATLAB_SMOOTHTS_SYNTHETIC_FIXTURE_EXPORTED';
fixture.behavior_data_loaded = false;
fixture.fly_data_loaded = false;
fixture.navigation_performance_used = false;
fixture.matlab_version = version;
fixture.matlab_release = version('-release');
fixture.smoothts_path = which('smoothts');

finance_info = ver('finance');
if isempty(finance_info)
    fixture.financial_toolbox = struct('name', '', 'version', '', 'release', '', 'date', '');
else
    fixture.financial_toolbox = struct( ...
        'name', finance_info(1).Name, ...
        'version', finance_info(1).Version, ...
        'release', finance_info(1).Release, ...
        'date', finance_info(1).Date);
end

fixture.period_length = 3;
fixture.alpha = 0.5;
fixture.alpha_formula = '2/(period_length+1)';

names = { ...
    'constant', ...
    'ramp', ...
    'impulse_first', ...
    'impulse_middle', ...
    'step', ...
    'sparse_counts' ...
};
inputs = { ...
    [1 1 1 1 1 1 1 1], ...
    [1 2 3 4 5 6 7 8], ...
    [1 0 0 0 0 0 0 0], ...
    [0 0 0 1 0 0 0 0], ...
    [0 0 0 1 1 1 1 1], ...
    [0 0 1 0 2 0 0 1] ...
};

fixtures = repmat(struct(), 1, numel(names));
max_period_alpha_difference = 0;
max_candidate_difference = 0;

for idx = 1:numel(names)
    x = double(inputs{idx});
    y_period = smoothts(x, 'e', 3);
    y_alpha = smoothts(x, 'e', 0.5);
    y_candidate = local_candidate_exponential(x, 0.5);

    fixtures(idx).name = names{idx};
    fixtures(idx).input = x;
    fixtures(idx).smoothts_period_3 = y_period;
    fixtures(idx).smoothts_alpha_0p5 = y_alpha;
    fixtures(idx).candidate_first_observation_recurrence = y_candidate;
    fixtures(idx).period_vs_alpha_max_abs_difference = max(abs(y_period - y_alpha));
    fixtures(idx).period_vs_candidate_max_abs_difference = max(abs(y_period - y_candidate));
    fixtures(idx).period_div_1p5 = y_period / 1.5;

    max_period_alpha_difference = max( ...
        max_period_alpha_difference, fixtures(idx).period_vs_alpha_max_abs_difference);
    max_candidate_difference = max( ...
        max_candidate_difference, fixtures(idx).period_vs_candidate_max_abs_difference);
end

fixture.fixtures = fixtures;
fixture.max_period_vs_alpha_abs_difference = max_period_alpha_difference;
fixture.max_period_vs_candidate_abs_difference = max_candidate_difference;
fixture.candidate_recurrence = [ ...
    's(1)=x(1); s(t)=alpha*x(t)+(1-alpha)*s(t-1), t>1' ...
];
fixture.integral_correction = struct( ...
    'historical_value', 1.5, ...
    'applied_to_fixture_outputs_for_inspection', true, ...
    'declared_part_of_final_figure3c_pipeline', false);
fixture.interpretation_boundary = [ ...
    'This fixture establishes installed smoothts numerical behavior only. ', ...
    'It does not select DNa02 spike thresholds, open yaw, compute Figure 3C, ', ...
    'or decide whether the historical /1.5 correction belongs to the final pipeline.' ...
];

encoded = jsonencode(fixture);
fid = fopen(output_path, 'w');
if fid < 0
    error('fly_sniff:fixture_write_failed', 'Could not open output path: %s', output_path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid, encoded, 'char');
fwrite(fid, sprintf('\n'), 'char');

fprintf('Wrote synthetic smoothts fixture: %s\n', output_path);
fprintf('max |period-3 - alpha-0.5| = %.17g\n', max_period_alpha_difference);
fprintf('max |smoothts - candidate recurrence| = %.17g\n', max_candidate_difference);
end


function y = local_candidate_exponential(x, alpha)
y = zeros(size(x));
y(1) = x(1);
for idx = 2:numel(x)
    y(idx) = alpha * x(idx) + (1 - alpha) * y(idx - 1);
end
end
