% Configure MATLAB to use the repository's canonical Python environment.
thisFile = mfilename("fullpath");
matlabDir = fileparts(thisFile);
repoRoot = fileparts(matlabDir);

if ispc
    pythonExe = fullfile(repoRoot, ".venv", "Scripts", "python.exe");
else
    pythonExe = fullfile(repoRoot, ".venv", "bin", "python");
end

if ~isfile(pythonExe)
    error("IPAT:PythonEnvMissing", ...
        "Canonical Python environment not found at %s. Run uv sync --dev first.", pythonExe);
end

pe = pyenv;
if pe.Status == "Loaded" && pe.Executable ~= string(pythonExe)
    if pe.ExecutionMode == "OutOfProcess"
        terminate(pe);
    else
        error("IPAT:PythonAlreadyLoaded", ...
            "MATLAB already loaded another in-process Python. Restart MATLAB and rerun this script.");
    end
end

pyenv(Version=pythonExe, ExecutionMode="OutOfProcess");
disp(pyenv);
