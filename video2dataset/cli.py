"""cli entry point"""

import fire

from video2dataset import video2dataset


def main():
    """Main entry point"""
    fire.Fire(video2dataset)


if __name__ == "__main__":
    main()
