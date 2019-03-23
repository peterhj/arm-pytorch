import guppy.v1 as gp

def main():
    mut_cache = gp.mutable_cache()
    mut_cache.fetch_once(
        "https://download.pytorch.org/whl/cu80/torch-0.3.1-cp35-cp35m-linux_x86_64.whl"
    )
    mut_cache.out()

    task = gp.task(name="train atari/arm", toolchain="python3")
    task.require_distro("ubuntu 16.04")
    task.require_cuda("8.0")

    # Install prerequisites.
    task.sh("apt-get install -y libglib2.0-0 libsm6 libxrender1 libfontconfig1 libxext6 cmake libboost-all-dev")
    task.sh("pip3 install numpy")
    task.sh("pip3 install opencv-python")
    task.sh("pip3 install /mutable_cache/torch-0.3.1-cp35-cp35m-linux_x86_64.whl")
    task.sh("cd gym")
    task.sh("pip3 install .[atari]")
    task.sh("cd ../doom-py")
    task.sh("pip3 install .")
    task.sh("cd ../gym-doom")
    task.sh("pip3 install .")
    task.sh("cd ..")

    # Do the training (will take some hours).
    task.sh("python3 ./train_doom_arm.py")

    run = gp.run()
    run.append(task)
    run.out()

if __name__ == "__main__":
    main()
