all: host device

device: qr.cu
	nvcc -arch=compute_35 -code=sm_35 -O3 -L/software/tamusc/magma/2.0.2-intel-2015B-CUDA-7.5.18/lib -lmagma -lmkl -I/software/tamusc/magma/2.0.2-intel-2015B-CUDA-7.5.18/include -ccbin=icc qr.cu -o qr_device.exe

host: qr.c
	icc -std=c99 -Wall -Wshadow -Wextra -g qr.c -o qr_host.exe -lm

clean:
	rm qr_device.exe
	rm qr_host.exe

