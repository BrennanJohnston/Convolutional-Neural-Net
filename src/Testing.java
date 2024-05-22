import java.util.Arrays;

public class Testing {

	public static void main(String[] args) {
		int[] layerWidths = {4, 4, 1};
		BrennanNeuralNetwork neuralNet = new BrennanNeuralNetwork(2, layerWidths, false);
		double[][] trainingData = new double[55][];
		double[][] trainingOuts = new double[55][];
		
		int epochs = 1000;
		//neuralNet.Train(epochs, 0.2, trainingData, trainingOuts);
		for(int e = 0; e < epochs; e++) {
			for(int i = 0; i < trainingData.length; i++) {
				neuralNet.ForwardPropagate(trainingData[i]);
				neuralNet.BackPropagate(0.2, trainingData[i], trainingOuts[i]);
			}
		}
		
		for(int i = 0; i < trainingData.length; i++) {
			double[] out = neuralNet.ForwardPropagate(trainingData[i]);
			System.out.println("Expected: " + Arrays.toString(trainingOuts[i]) + " --- Output: " + Arrays.toString(out));
		}
	}
}