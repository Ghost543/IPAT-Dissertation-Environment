% Capture MATLAB-side environment information.
thisFile = mfilename("fullpath");
matlabDir = fileparts(thisFile);
repoRoot = fileparts(matlabDir);

fprintf("Environment ID: IPAT-DISS-ENV-v1\n");
fprintf("MATLAB release: %s\n", version("-release"));
fprintf("MATLAB version: %s\n", version);
fprintf("Computer: %s\n", computer);

fprintf("\nInstalled products:\n");
v = ver;
for k = 1:numel(v)
    fprintf("  %s | %s | %s\n", v(k).Name, v(k).Version, v(k).Release);
end

fprintf("\nPython environment visible to MATLAB:\n");
try
    run(fullfile(matlabDir, "configure_python.m"));
catch ME
    fprintf(2, "Python configuration check failed: %s\n", ME.message);
end

outDir = fullfile(repoRoot, "env", "manifests");
if ~isfolder(outDir)
    mkdir(outDir);
end
fid = fopen(fullfile(outDir, "matlab_environment.txt"), "w");
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "Environment ID: IPAT-DISS-ENV-v1\n");
fprintf(fid, "MATLAB release: %s\n", version("-release"));
fprintf(fid, "MATLAB version: %s\n", version);
fprintf(fid, "Computer: %s\n", computer);
for k = 1:numel(v)
    fprintf(fid, "%s | %s | %s\n", v(k).Name, v(k).Version, v(k).Release);
end
