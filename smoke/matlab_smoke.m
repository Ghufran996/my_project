% MATLAB smoke: version, licences, COMSOL LiveLink availability.
root = getenv('H2SU_ROOT'); outfile = fullfile(root, 'results', 'smoke', 'matlab_smoke.txt');
fid = fopen(outfile, 'w');
fprintf(fid, 'MATLAB version: %s\n', version);
fprintf(fid, 'date: %s\n', datestr(now,'yyyy-mm-dd HH:MM:SS'));
feats = {'MATLAB','Simulink','Simscape','Optimization_Toolbox','Statistics_Toolbox','Distrib_Computing_Toolbox','GADS_Toolbox','Control_Toolbox','MPC_Toolbox'};
for k = 1:numel(feats)
    try
        t = license('test', feats{k});
        c = license('checkout', feats{k});
    catch ME
        t = -1; c = -1; fprintf(fid, '  error %s: %s\n', feats{k}, ME.message);
    end
    fprintf(fid, 'license %-28s test=%d checkout=%d -> %s\n', feats{k}, t, c, tf(t==1 && c==1));
end
fprintf(fid, '--- ver ---\n');
v = ver;
for k = 1:numel(v)
    fprintf(fid, '%s | %s\n', v(k).Name, v(k).Version);
end
fprintf(fid, '--- LiveLink ---\n');
cpath = 'C:\Program Files\COMSOL\COMSOL62\Multiphysics\mli';
fprintf(fid, 'mli folder exists: %d\n', exist(cpath,'dir')==7);
try
    addpath(cpath);
    mphstart;   % starts COMSOL server on default port
    import com.comsol.model.*
    import com.comsol.model.util.*
    fprintf(fid, 'mphstart: OK\n');
    fprintf(fid, 'mphversion: %s\n', char(ModelUtil.getComsolVersion));
    fprintf(fid, 'LiveLink MATLAB: TRUE\n');
catch ME
    fprintf(fid, 'LiveLink MATLAB: FALSE\n error id: %s\n error: %s\n', ME.identifier, ME.message);
end
fclose(fid);
function s = tf(b)
if b, s = 'TRUE'; else, s = 'FALSE'; end
end
