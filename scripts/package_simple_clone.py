import os
import json
import shutil

def package_clone(mesh_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Copy all mesh JSON files
    mesh_files = [f for f in sorted(os.listdir(mesh_dir)) if f.endswith(".json")]
    for f in mesh_files:
        shutil.copy(os.path.join(mesh_dir, f), os.path.join(output_dir, f))

    # Create a basic clone config file
    config = {
        "name": "SimpleClone",
        "frames": len(mesh_files),
        "mesh_files": mesh_files,
        "texture": None,  
        "description": "Basic static clone without audio"
    }

    with open(os.path.join(output_dir, "clone_config.json"), "w") as cf:
        json.dump(config, cf, indent=4)

    print(f"Packaged simple clone with {len(mesh_files)} frames at {output_dir}")

if __name__ == "__main__":
    mesh_folder = "../face_mesh"
    output_folder = "../clone_package"
    package_clone(mesh_folder, output_folder)
