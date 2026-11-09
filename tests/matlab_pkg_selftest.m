% MATLAB_PKG_SELFTEST  Non-COMSOL logic test of the COMSOL_MATLAB helpers (parameter reader, event definitions, CSV export).
%   Uses a SYNTHETIC breakthrough curve; the output goes to the folder in H2SU_SELFTEST_OUT and is never a result.
root = getenv('H2SU_ROOT');
outdir = getenv('H2SU_SELFTEST_OUT');
addpath(fullfile(root, 'COMSOL_MATLAB'));
if ~isfolder(outdir), mkdir(outdir); end
lines = {};
% 1 parameter reader
p = read_params(fullfile(root, 'COMSOL_MATLAB', 'params_comsol.csv'), 'ver_breakthrough');
lines{end+1} = sprintf('read_params: K_ads=%g L_m=%g R_gas=%.12g n_meta=%d', p.K_ads, p.L_m, p.R_gas_J_mol_K, numel(p.meta_));
try
    require_params(p, {'rate_A_SI'}, 'SN18', 'selftest');
    lines{end+1} = 'require_params: NO ERROR (unexpected)';
catch err
    lines{end+1} = ['require_params error text OK: ' strrep(err.message, newline, ' | ')];
end
c = read_cases(fullfile(root, 'COMSOL_MATLAB', 'cases_comsol.csv'));
lines{end+1} = sprintf('read_cases: %d cases, status = %s', height(c), strjoin(cellstr(c.status), ','));
% 2 event definitions on a synthetic breakthrough
t = 0:0.5:40;
ratio = 0.5 * (1 + erf((t - 17) / 2));
res.t = t;
res.series = struct('y_O2_out', ratio);
ev0 = make_evcfg(p);
ev0.break_col = 'y_O2_out'; ev0.y_in = 1;
res.evcfg = ev0;
res.ev = compute_events(res);
lines{end+1} = sprintf('compute_events: t_break=%.6f (synthetic erf front centred at 17 s; expected 17)', res.ev.t_break_s);
lines{end+1} = sprintf('last_crossing_time: %.6f (r = 2 - t/10 on t=0:0.5:40 ... expected 10)', last_crossing_time(0:0.5:40, 2 - (0:0.5:40)/10));
lines{end+1} = sprintf('trapz_to: %.6f (integral of t from 0 to 3.25 = 5.28125)', trapz_to(0:0.5:5, 0:0.5:5, 3.25));
res.info = struct('wall_s', 1.5);
export_ver6_csv(res, 'SYNTHETIC_selftest', 'N400', 1e-6, outdir);
lines{end+1} = 'export_ver6_csv: written';
fid = fopen(fullfile(outdir, 'selftest_log.txt'), 'w');
fprintf(fid, '%s\n', lines{:});
fclose(fid);
fprintf('%s\n', lines{:});
