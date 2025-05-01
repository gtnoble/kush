module app;

import io.cli;

void main(string[] args) {
    auto options = parseCommandLine(args);
    auto runner = SimulationRunner(options);
    runner.run();
}
