import guppy.v1 as gp

def main():
    mut_cache = gp.mutable_cache()
    mut_cache.fetch_once(
        #"https://download.pytorch.org/whl/cu80/torch-0.3.1-cp35-cp35m-linux_x86_64.whl"
        "https://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27m-linux_x86_64.whl"
    )
    mut_cache.out()

    run = gp.run()

    for env_id in ["ppaquette/DoomCorridor-v0", "ppaquette/DoomMyWayHome-v0"]:
        task = gp.task(name="train doom/arm ({})".format(env_id), toolchain="python2")
        task.require_distro("ubuntu 16.04")
        task.require_cuda("8.0")
        # NOTE: bash is now the default shell.
        #task.require_shell("bash")

        # Install prerequisites.
        task.sh("apt-get install -y libglib2.0-0 libsm6 libxrender1 libfontconfig1 libxext6 cmake libboost-all-dev libsdl2-dev wget unzip")
        task.sh("pip install numpy")
        task.sh("pip install opencv-python")
        task.sh("pip install /mutable_cache/torch-0.3.1-cp27-cp27m-linux_x86_64.whl")
        task.sh("cd gym")
        task.sh("pip install .[atari]")
        task.sh("cd ../doom-py")
        task.sh("pip install .")
        task.sh("cd ../gym-doom")
        task.sh("pip install .")
        task.sh("cd ..")

        # Do the training (will take some hours).
        task.sh("python ./train_doom_arm.py {}".format(env_id))

        run.append(task)

    run.out()

if __name__ == "__main__":
    main()
