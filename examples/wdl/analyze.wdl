## Run `proteus analyze` on a structure through any GA4GH TES server.
## With Sprocket:  sprocket run -c examples/wdl/sprocket.toml examples/wdl/analyze.wdl examples/wdl/inputs.json
version 1.2

task analyze {
    meta {
        description: "All-atom biophysics of a PDB/mmCIF file, executed in the proteus container"
    }
    input {
        File structure
        String image = "ghcr.io/otoyuki/proteus:latest"
    }
    command <<<
        proteus analyze --pdb ~{structure} | tee metrics.txt
    >>>
    output {
        File metrics = "metrics.txt"
    }
    requirements {
        container: image
        cpu: 1
        memory: "1 GiB"
    }
}

workflow analyze_structure {
    input {
        File structure
        String image = "ghcr.io/otoyuki/proteus:latest"
    }
    call analyze { input: structure = structure, image = image }
    output {
        File metrics = analyze.metrics
    }
}
