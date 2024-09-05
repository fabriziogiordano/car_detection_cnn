To compress the executable created by PyInstaller, you can use the UPX (Ultimate Packer for eXecutables) compression tool. UPX is an open-source tool that compresses executables, making them smaller.

### Using UPX with PyInstaller

PyInstaller has built-in support for UPX, but you need to have UPX installed on your system to use it. Here’s how you can integrate UPX with PyInstaller:

#### Step 1: Install UPX

- **On Windows**:
  Download UPX from the [official UPX website](https://upx.github.io/). Extract the UPX executable and add its location to your system PATH.

- **On macOS** (using Homebrew):
  ```bash
  brew install upx
  ```

- **On Linux**:
  Most distributions include UPX in their package repositories. For example, on Debian-based systems (like Ubuntu):
  ```bash
  sudo apt-get install upx-ucl
  ```

#### Step 2: Modify PyInstaller Command

To use UPX compression with PyInstaller, you just need to ensure that UPX is installed and accessible in your PATH. PyInstaller will automatically use UPX if it is available.

Run PyInstaller with the `--upx-dir` option to specify the directory containing the UPX executable (if it is not in your PATH), or just run PyInstaller normally if UPX is in your PATH.

Here’s the command to build the executable with UPX compression:

```bash
pyinstaller --onefile --upx-dir /path/to/upx run_inference_pt_percentages.py
```

Replace `/path/to/upx` with the actual path to your UPX executable if it's not in your PATH. If UPX is in your PATH, you can omit the `--upx-dir` option:

```bash
pyinstaller --onefile run_inference_pt_percentages.py
```

### Notes:

1. **Compression Levels**: UPX provides different compression levels. The default compression is usually sufficient, but if you need more control, you can use UPX directly with command-line options:
   ```bash
   upx --best --lzma your_executable
   ```
   This compresses the executable with the best compression ratio and LZMA algorithm.

2. **Executable Size**: Compression can significantly reduce the size of the executable, but it might slightly increase the startup time as the executable needs to be decompressed in memory.

3. **Compatibility**: While UPX compression generally works well, it can sometimes cause issues with certain executables. Always test the compressed executable to ensure it runs correctly.

By following these steps, you can reduce the size of your PyInstaller-built executable and make it more efficient for distribution and deployment.