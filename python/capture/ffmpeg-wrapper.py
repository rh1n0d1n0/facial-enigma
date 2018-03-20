import os
import time
import subprocess

class Ffmpeg(object):
    """FFMPEG wrapper using the python subprocess module"""

    def __init__(self, input_ , output, filters=[], ffmpeg='ffmpeg'):
        self.in_args = input_
        self.out_args = output
        self.filters = filters
        self.bin = ffmpeg
        self.cmd = [arg for args in [self.in_args, self.filters, self.out_args] for arg in args]
        self.cmd.insert(0, self.bin)

    def execute(self):
        self.proc = subprocess.Popen(
            self.cmd,
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            check = True,
        )
        while True:
            self.output = self.proc.stdout.readline()
            if output == '' and process.poll() is not None:
                break
       return process.poll()

