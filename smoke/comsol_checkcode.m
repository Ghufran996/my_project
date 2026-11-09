% COMSOL_CHECKCODE  Static check (checkcode) of every .m file in COMSOL_MATLAB. No COMSOL is started.
%   Run:  matlab -batch "run('code/smoke/comsol_checkcode.m')"   (from the project root)
root = getenv('H2SU_ROOT');
if isempty(root)
    error('Set H2SU_ROOT to the project root.');
end
pkg = fullfile(root, 'COMSOL_MATLAB');
outfile = fullfile(root, 'results', 'smoke', 'comsol_checkcode.txt');
files = dir(fullfile(pkg, '*.m'));
fid = fopen(outfile, 'w');
fprintf(fid, 'checkcode of COMSOL_MATLAB/*.m, MATLAB %s, %s\n', version, char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')));
fprintf(fid, 'Static analysis only; COMSOL was NOT started and no model was built or solved by the agent.\n\n');
n_err = 0; n_msg = 0;
for i = 1:numel(files)
    f = fullfile(pkg, files(i).name);
    info = checkcode(f, '-struct', '-id');
    fprintf(fid, '%s : %d message(s)\n', files(i).name, numel(info));
    for k = 1:numel(info)
        fprintf(fid, '   line %d: [%s] %s\n', info(k).line, info(k).id, info(k).message);
        n_msg = n_msg + 1;
        if startsWith(info(k).id, 'SYN') || contains(info(k).message, 'Parse error') || contains(info(k).message, 'Invalid syntax')
            n_err = n_err + 1;
        end
    end
end
fprintf(fid, '\nTOTAL: %d files, %d messages, %d syntax errors\n', numel(files), n_msg, n_err);
fclose(fid);
fprintf('checkcode done: %d files, %d messages, %d syntax errors\n', numel(files), n_msg, n_err);
