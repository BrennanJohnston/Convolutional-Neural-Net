import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

public class PS5 {
	private static double[][] xdata, xdataTraining, xdataTesting, w1data, w2data, ydataConverted;
	private static int[] ydata, ydataTesting;
	private final static double TRAINING_DATA_RATIO = 0.8;
	
	public static void main(String[] args) {
		if(args.length < 4) {
			System.out.println("Incorrect number of arguments, expected 4.");
			return;
		}
		
		String w1path, w2path, xdatapath, ydatapath;
		w1path = args[0];
		w2path = args[1];
		xdatapath = args[2];
		ydatapath = args[3];
		
		File w1File, w2File, xdataFile, ydataFile;
		w1File = new File(w1path);
		w2File = new File(w2path);
		xdataFile = new File(xdatapath);
		ydataFile = new File(ydatapath);
		
		if(!w1File.exists() || w1File.isDirectory()) {
			System.out.println("Weight file 1 either does not exist or is a directory.  Aborting.");
			return;
		}
		
		if(!w2File.exists() || w2File.isDirectory()) {
			System.out.println("Weight file 2 either does not exist or is a directory.  Aborting.");
			return;
		}
		
		if(!xdataFile.exists() || xdataFile.isDirectory()) {
			System.out.println("X data file either does not exist or is a directory.  Aborting.");
			return;
		}
		
		if(!ydataFile.exists() || ydataFile.isDirectory()) {
			System.out.println("Y data file either does not exist or is a directory.  Aborting.");
			return;
		}
		
		try {
			xdata = parseData(xdatapath);
			ydata = parseDatay(ydatapath);
			
			w1data = parseData(w1path);
			w2data = parseData(w2path);
			
			//convert ydata ints to arrays of length 10 which indicate desired neural net output
			ydataConverted = new double[ydata.length][];
			for(int i = 0; i < ydataConverted.length; i++) {
				ydataConverted[i] = new double[10]; //java creates arrays of double initialized to 0.0 on all indexes
				int index = ydata[i] - 1;
				if(index < 0) index = 9;
				ydataConverted[i][index] = 1.0;
				
			}
			
			//parse out xdata into a training and testing set
			int trainingSetCount = (int)Math.floor(xdata.length * TRAINING_DATA_RATIO);
			xdataTraining = new double[trainingSetCount][];
			xdataTesting = new double[xdata.length - trainingSetCount][];
			for(int x = 0; x < xdataTraining.length; x++) {
				xdataTraining[x] = Arrays.copyOf(xdata[x], xdata[x].length);
			}
			for(int x = 0; x < xdataTesting.length; x++) {
				xdataTesting[x] = Arrays.copyOf(xdata[xdataTraining.length + x], xdata[xdataTraining.length + x].length);
			}
			
			ydataTesting = new int[ydata.length - trainingSetCount];
			for(int y = 0; y < ydataTesting.length; y++) {
				ydataTesting[y] = ydata[trainingSetCount + y];
			}
			
			//end parsing ==========================================
			
			//create neural net ====================================
			int[] layerWidths = {30, 10};
			BrennanNeuralNetwork neuralNet = new BrennanNeuralNetwork(784, layerWidths, true);
			//set network weights
			for(int n = 0; n < neuralNet.LAYER_WIDTHS[0]; n++) {
				//neuralNet.setWeights(0, n, w1data[n]);
			}
			for(int n = 0; n < neuralNet.LAYER_WIDTHS[1]; n++) {
				//neuralNet.setWeights(1, n, w2data[n]);
			}

			System.out.println("************************************************************\r\n"
					+ "Problem Set: Problem Set 5: Neural Network\r\n"
					+ "Name: Brennan Johnston\r\n"
					+ "Synax: java PS5 w1.txt w2.txt xdata.txt ydata.txt\r\n"
					+ "************************************************************\r\n\n");
			
			System.out.printf("Training Phase:  %s\n", xdatapath);
			System.out.printf("------------------------------------------------------------------------------------------\n"
					+ "   => Number of entries (n):%15s\n"
					+ "   => Number of Features (p):%14s\n\n\n"
					+ "Starting Gradient Descent:\n"
					+ "------------------------------------------------------------------------------------------\n\n", String.valueOf(xdata.length), String.valueOf(xdata[0].length));
			
			//train neural net for 700 epochs
			neuralNet.TrainWithOutput(15, 0.10, 0.25, xdataTraining, ydataConverted);

			//check accuracy
			int accuracyIterationCount = Math.min(2000, xdataTesting.length), correctCount = 0;
			boolean correct;
			System.out.printf("\nTesting phase (first %d records):\n"
					+ "------------------------------------------------------------------------------------------\n\n", accuracyIterationCount);

			for(int i = 0; i < accuracyIterationCount; i++) {
				double[] out = neuralNet.ForwardPropagate(xdataTesting[i]);
				int indexOfLargest = 0;
				correct = false;
				for(int k = 1; k < out.length; k++) {
					if(out[k] > out[indexOfLargest])
						indexOfLargest = k;
				}
				indexOfLargest++;
				if(indexOfLargest == ydataTesting[i]) { correct = true; correctCount++; }
				
				System.out.printf("    Test Record %d:%3d    Prediction:%3d    Correct:%3s\n", i+1, ydata[xdataTraining.length + i], indexOfLargest, String.valueOf(correct));
			}
			
			System.out.printf("    => Number of test entries (n):%10d\n"
					+ "    => Accuracy:%27.0f%%\n", accuracyIterationCount, ((double)correctCount / accuracyIterationCount) * 100);
			
			//print weights to file w1out and w2out
			double[][][] finalWeights = neuralNet.getWeights();
			for(int l = 0; l < finalWeights.length; l++) {
				String s = "";
				for(int n = 0; n < finalWeights[l].length; n++) {
					for(int w = 0; w < finalWeights[l][n].length; w++) {
						s += String.format("%.4f", finalWeights[l][n][w]);
						if(w < finalWeights[l][n].length-1) s+=",";
					}
					if(n < finalWeights[l].length-1) s+="\n";
				}
				FileWriter fw = new FileWriter("w"+(l+1)+"out.txt");
				fw.write(s);
				fw.close();
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static double sigmoid(double x) {
		return 1 / (1 + Math.pow(Math.E, -x));
	}
	
	private static double sigmoidDerivative(double x) {
		return sigmoid(x)*(1 - sigmoid(x));
	}
	
	private static double sigmoidDerivativeWithGX(double gx) {
		return gx*(1 - gx);
	}
	
	private static int[] parseDatay(String path) throws IOException {
		int[] data = null;
		int lineCount = 0;
		String line, lineSplit[];
		FileInputStream fis = new FileInputStream(path);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis));
		while((line = br.readLine()) != null) {
			lineCount++;
		}
		
		data = new int[lineCount];
		lineCount = 0;
		fis.getChannel().position(0);
		br = new BufferedReader(new InputStreamReader(fis));
		while((line = br.readLine()) != null) {
			data[lineCount] = Integer.parseInt(line);
			lineCount++;
		}
		
		br.close();
		return data;
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
