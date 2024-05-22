import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

public class BrennanNeuralNetwork {	
	private Layer[] network; /*layer, neuron*/
	private final boolean ACTIVATE_OUTPUT;
	public final int LAYER_COUNT, LAYER_WIDTHS[];
	
	public BrennanNeuralNetwork(int inputCount, int[] layerWidths, boolean activateOutput) {
		ACTIVATE_OUTPUT = activateOutput;
		LAYER_COUNT = layerWidths.length;
		LAYER_WIDTHS = layerWidths;
		network = new Layer[layerWidths.length];
		
		for(int l = 0; l < network.length; l++) {
			network[l] = new Layer(LAYER_WIDTHS[l], inputCount);
			for(int n = 0; n < network[l].NEURON_COUNT; n++) {
				for(int w = 0; w < network[l].weights[n].length; w++) {
					//apply random weights from -1 to 1
					network[l].weights[n][w] = Math.random() * 2 - 1;
				}
			}
			
			inputCount = network[l].NEURON_COUNT;
		}
	}
	
	public void TrainWithOutput(int epochs, double learningRate, double lambda, double[][] trainingData, double[][] expectedOutputs) {
		double totalLossSum, sum, loss;
		int numCorrect, longestOutIndex, longestExpectedIndex;
		Double previousLoss = null, delta = null;
		String lossString = "", accuracyString = "";
		
		for(int e = 0; e < epochs; e++) {
			totalLossSum = 0.0;
			numCorrect = 0;
			for(int t = 0; t < trainingData.length; t++) {
				double[] out = ForwardPropagate(trainingData[t]);
				BackPropagate(learningRate, trainingData[t], expectedOutputs[t]);
				
				//check if correct
				longestOutIndex = 0;
				longestExpectedIndex = 0;
				for(int i = 1; i < out.length; i++)
					if(out[i] > out[longestOutIndex]) longestOutIndex = i;
				longestOutIndex++;
				for(int i = 1; i < expectedOutputs[t].length; i++)
					if(expectedOutputs[t][i] > expectedOutputs[t][longestExpectedIndex]) longestExpectedIndex = i;
				longestExpectedIndex++;
				if(longestOutIndex == longestExpectedIndex) numCorrect++;
				
				//calculate loss inner summation
				sum = 0.0;
				for(int o = 0; o < out.length; o++) {
					sum += (-expectedOutputs[t][o]*Math.log(out[o])) - ((1-expectedOutputs[t][o])*Math.log(1-out[o]));
				}
				
				//outer summation
				totalLossSum += sum;
			}
			
			loss = totalLossSum / trainingData.length;
			
			//sum all weights squared
			double weightSquaredSum = 0.0;
			for(int l = 0; l < network.length; l++) {
				for(int n = 0; n < network[l].weights.length; n++) {
					for(int w = 0; w < network[l].weights[n].length; w++) {
						weightSquaredSum += Math.pow(network[l].weights[n][w], 2);
					}
				}
			}
			
			weightSquaredSum *= (lambda)/(2 * trainingData.length);
			
			loss += weightSquaredSum;
			
			//calculate delta
			if(previousLoss != null) {
				delta = (Math.abs(previousLoss - loss) * 100) / previousLoss;
			}
			
			previousLoss = loss;

			System.out.printf("Epoch %d:   Loss of %.2f     Delta = %.0f%%     Epsilon = %.0f%%\n", e+1, loss, delta, 0.03 * 100);
			
			lossString += String.format("%d, %.4f\n", e+1, loss);
			accuracyString += String.format("%d, %.3f%%\n", e+1, ((double)numCorrect / trainingData.length)*100);
		}
		
		FileWriter fw;
		try {
			fw = new FileWriter("loss.txt");
			fw.write(lossString);
			fw.close();
			fw = new FileWriter("accuracy.txt");
			fw.write(accuracyString);
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void Train(int epochs, double learningRate, double[][] trainingData, double[][] expectedOutputs) {
		for(int e = 0; e < epochs; e++) {
			for(int t = 0; t < trainingData.length; t++) {
				ForwardPropagate(trainingData[t]);
				BackPropagate(learningRate, trainingData[t], expectedOutputs[t]);
			}
		}
	}
	
	public double[] ForwardPropagate(double[] input) { //input does not include bias node
		double sum;
		for(int l = 0; l < network.length; l++) { //iterate each layer from first hidden layer
			for(int n = 0; n < network[l].NEURON_COUNT; n++) {
				sum = 0.0;
				for(int w = 0; w < input.length; w++) { //iterate all weight/input pairs, bias after
					sum += network[l].weights[n][w] * input[w];
				}
				sum += network[l].weights[n][network[l].weights[n].length-1] * 1; //bias calculation
				
				network[l].h[n] = sum;
				network[l].output[n] = sum;
				
				//check if at output layer and if we activate the output
				if(ACTIVATE_OUTPUT || l < network.length-1) {
					network[l].output[n] = sigmoid(network[l].h[n]);
				}
			}
			
			//set input to the outputs of the current layer, for the next layer to calculate with as input
			input = network[l].output;
		}
		
		//return the outputs of the last layer
		return network[network.length-1].output;
	}
	
	public void BackPropagate(double learningRate, double[] inputs, double[] expectedOutputs) { //inputs does not include bias
		//calculate output layer error
		for(int n = 0; n < network[network.length-1].NEURON_COUNT; n++) {
			network[network.length-1].delta[n] = expectedOutputs[n] - network[network.length-1].output[n];
			//do not multiply by derivative at output layer
		}
		
		//hidden layer error deltas
		for(int l = network.length-2; l >= 0; l--) {
			//iterate each neuron
			for(int n = 0; n < network[l].NEURON_COUNT; n++) {
				network[l].delta[n] = 0.0;
				//summation of weights times deltas
				for(int n1 = 0; n1 < network[l+1].NEURON_COUNT; n1++) {
					network[l].delta[n] += network[l+1].weights[n1][n] * network[l+1].delta[n1];
				}
				//multiply summation by g'(h_i)
				network[l].delta[n] *= sigmoidDerivative(network[l].h[n]);
			}
		}
		
		//adjust weights
		//iterate each layer from first hidden
		for(int l = 0; l < network.length; l++) {
			//iterate each neuron
			for(int n = 0; n < network[l].NEURON_COUNT; n++) {
				//iterate each weight and adjust
				for(int w = 0; w < inputs.length; w++) {
					network[l].weights[n][w] += learningRate * inputs[w] * network[l].delta[n];
				}
				//bias adjustment
				network[l].weights[n][network[l].weights[n].length-1] += learningRate * 1 * network[l].delta[n];
			}
			
			//set input to the outputs of the current layer, for the next layer to calculate with as input
			inputs = network[l].output;
		}
	}
	
	public double[][][] getWeights() {
		double[][][] weights = new double[LAYER_COUNT][][];
		for(int l = 0; l < weights.length; l++) {
			weights[l] = new double[network[l].NEURON_COUNT][];
			for(int n = 0; n < weights[l].length; n++) {
				weights[l][n] = Arrays.copyOf(network[l].weights[n], network[l].weights[n].length);
			}
		}
		
		return weights;
	}
	
	//weights provided must include a weight at the end for bias
	public void setWeights(int layerNum, int neuronNum, double[] weights) {
		if(layerNum < 0 || layerNum > network.length-1 || neuronNum < 0 || neuronNum > network[layerNum].NEURON_COUNT) {
			System.out.println("setWeights(): layerNum or neuronNum out of bounds.");
			return;
		}
		
		network[layerNum].weights[neuronNum] = weights;
	}
	
	private static double sigmoid(double x) {
		return 1 / (1 + Math.pow(Math.E, -x));
	}
	
	private static double sigmoidDerivative(double x) {
		return sigmoid(x)*(1 - sigmoid(x));
	}
	
	public class Layer {
		public double[] weights[], h, output, delta;
		public final int NEURON_COUNT;
		
		public Layer(int wid, int inputCount) {
			NEURON_COUNT = wid;
			weights = new double[NEURON_COUNT][inputCount + 1]; //+1 for bias
			h = new double[NEURON_COUNT];
			output = new double[NEURON_COUNT];
			delta = new double[NEURON_COUNT];
		}
	}
}