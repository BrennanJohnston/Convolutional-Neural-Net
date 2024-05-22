import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

import javax.imageio.ImageIO;

public class ImageProcessingTesting {

	private static final int IMAGE_WIDTH = 28, IMAGE_COUNT = 10;
	
	public static void main(String[] args) {
		if(args.length < 3) {
			System.out.println("Invalid number of arguments, expected 3.");
		}
		
		String imagepath, w1path, w2path;
		imagepath = args[0];
		w1path = args[1];
		w2path = args[2];
		
		try {
			BufferedImage numsImage = ImageIO.read(new File(imagepath));
			BufferedImage[] images = new BufferedImage[IMAGE_COUNT];
			if(numsImage.getWidth() < IMAGE_WIDTH * IMAGE_COUNT) {
				System.out.println("Image size invalid.");
				return;
			}
			
			int offset = 0;
			for(int i = 0; i < IMAGE_COUNT; i++) {
				int topleftx = i + (offset * IMAGE_WIDTH), bottomrightx = topleftx + IMAGE_WIDTH;
				images[i] = numsImage.getSubimage(topleftx, 0, bottomrightx, IMAGE_WIDTH);
			}
			
			double[][] inputData = new double[IMAGE_COUNT][];
			double[][] outData   = new double[IMAGE_COUNT][];
			
			//create x data
			for(int i = 0; i < inputData.length; i++) {
				double[] data = new double[IMAGE_WIDTH*IMAGE_WIDTH];
				int index = 0;
				//((DataBufferByte)images[i].getRaster().getDataBuffer()).getData();
				for(int x = 0; x < IMAGE_WIDTH; x++) {
					for(int y = 0; y < IMAGE_WIDTH; y++) {
						int c = images[i].getRGB(x, y);
						int b = c & 0xff;
						int g = (c & 0xff00) >> 8;
						int r = (c & 0xff0000) >> 16;
						double greyscale = (r+g+b)/3;
						data[index] = 1 - greyscale / 255;
						index++;
					}
				}
				
				inputData[i] = data;
			}
			
			//create y data
			for(int i = 0; i < outData.length; i++) {
				outData[i] = new double[IMAGE_COUNT];
				if(i == 0)
					outData[i][outData.length-1] = 1.0;
				else
					outData[i][i-1] = 1.0;
			}
			
			double[][] w1data = parseData(w1path);
			double[][] w2data = parseData(w2path);
			
			int[] layerWidths = {30, 10};
			BrennanNeuralNetwork neuralNet = new BrennanNeuralNetwork(784, layerWidths, false);
			for(int n = 0; n < neuralNet.LAYER_WIDTHS[0]; n++) {
				neuralNet.setWeights(0, n, w1data[n]);
			}
			for(int n = 0; n < neuralNet.LAYER_WIDTHS[1]; n++) {
				neuralNet.setWeights(1, n, w2data[n]);
			}
			
			boolean correct;
			for(int i = 0; i < inputData.length; i++) {
				double[] out = neuralNet.ForwardPropagate(inputData[i]);
				int indexOfLargest = 0, indexOfLargest2 = 0;
				correct = false;
				for(int k = 1; k < out.length; k++) {
					if(out[k] > out[indexOfLargest])
						indexOfLargest = k;
				}
				indexOfLargest++;
				for(int k = 1; k < outData[i].length; k++) {
					if(outData[i][k] > outData[i][indexOfLargest2])
						indexOfLargest2 = k;
				}
				indexOfLargest2++;
				
				if(indexOfLargest == indexOfLargest2) { correct = true; }
				
				//System.out.println("============================================");
				System.out.printf("Expected: %d    Output: %d    Correct: %s\n", indexOfLargest2, indexOfLargest, String.valueOf(correct));
				//System.out.println(Arrays.toString(out));
				//System.out.println(Arrays.toString(outData[i]));
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static double[][] parseData(String path) throws IOException {
		double[][] data = null;
		int lineCount = 0;
		String line, lineSplit[];
		FileInputStream fis = new FileInputStream(path);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis));
		while((line = br.readLine()) != null) {
			lineCount++;
		}

		data = new double[lineCount][];
		lineCount = 0;
		fis.getChannel().position(0);
		br = new BufferedReader(new InputStreamReader(fis));
		while((line = br.readLine()) != null) {
			lineSplit = line.split(",");
			data[lineCount] = new double[lineSplit.length];
			System.arraycopy(Arrays.stream(lineSplit).mapToDouble(Double::parseDouble).toArray(), 0, data[lineCount], 0, lineSplit.length-1);
			lineCount++;
		}
		
		br.close();
		
		return data;
	}
}
