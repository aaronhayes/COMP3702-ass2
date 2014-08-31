package machl;
import java.util.*;
import java.io.Serializable;
import java.math.*;

/**
 * A implementation of a multi-layers nerual network.
 */


/* Here's a list of things one can do:
 *  1. implement batched (epoch-based) weight update
 *  2. add "momentum" to weight adjustments
 *  3. enable more node layers (hidden layers).
 *  4. try alternative output functions (e.g. hyperbolic tangens, or softmax)
 *  5. try alternative error measures for gradient calculations (maximum likelihood)
 *  6. after a hidden layer has been added, a simple recurrent network can be constructed (see Elman, 1990)
 */

public class MLNN implements Serializable {
  public double input[];
  public double output[];
  
  public double[][] w;    // the trainable weight values [to node][from input]
  private double[][] w2;
  private double[][] w3;
  
  public double[] bias;   // the trainable bias values for nodes
  private double[] bias2;
  private double[] bias3;
  
  private double[] error;
  private double[] error2;
  private double[] error3;
  
  private int in;
  private int h1;
  private int h2;
  private int out;
  
  public double[] hidden1;
  public double[] hidden2;
  
  Random rand;            // a random number generator for initial weight values

  /** Constructs a neural network structure and initializes weights to small random values.
   *  @param  nInput  Number of input nodes
   *  @param  nOutput Number of output nodes
   *  @param  num_h1  Number of nodes in Hidden Layer 1
   *  @param  num_h2  Number of nodes in Hidden Layer 2
   *  @param  seed    Seed for the random number generator used for initial weights.
   *
   */
  public MLNN(int nInput, int nOutput, int num_h1, int num_h2, int seed) {

	in = nInput;
	out = nOutput;
	h1 = num_h1;
	h2 = num_h2;
	
    // allocate space for node and weight values
    input = new double[in];
	output = new double[out];
    hidden1 = new double[h1];
    hidden2 = new double[h2];
    
    w = new double[h1][in];
    w2 = new double[h2][h1];
    w3 = new double[out][h2];
    
    bias = new double[h1];
    bias2 = new double[h2];
    bias3 = new double[out];
    
    getWeights(seed);
    
    /* create arrays for errors */
    error = new double[out];
    error2 = new double[h1];
    error3 = new double[h2];
    
  }
  
  /**
   * Generate random weights
   * @param seed Seed for the random number generator used for initial weights.
   */
  public void getWeights(int seed) {
	  // Randomly get weights
	  Random rand = new Random(seed);
	  for (int j = 0; j < h1; j++) {
		  for (int i = 0; i < in; i++) {
			  w[j][i] = rand.nextGaussian()*.1;
		  }
		  bias[j]=rand.nextGaussian()*.1;
	  }
	  
	  rand = new Random(seed);
	  for (int j = 0; j < h2; j++) {
		  for (int i = 0; i < h1; i++) {
			  w2[j][i] = rand.nextGaussian()*.1;
		  }
		  bias2[j]=rand.nextGaussian()*.1;
	  }
	  
	  rand = new Random(seed);
	  for (int j = 0; j < out; j++) {
		  for (int i = 0; i < h2; i++) {
			  w3[j][i] = rand.nextGaussian()*.1;
		  }
		  bias3[j]=rand.nextGaussian()*.1;
	  }
  }

  /** The so-called output function. Computes the output value of a node 
   * 	given the summed incoming activation.
   *  You can use anyone you like if it is differentiable.
   *  This one is called the logistic function (a sigmoid) and produces 
   *  	values bounded between 0 and 1.
   *  @param  net The summed incoming activation
   *  @return double
   */
  public double outputFunction(double net) {
    return 1.0/(1.0+Math.exp(-net));
  }

  /** The derivative of the output function.
   *  This one is the derivative of the logistic function which is efficiently computed with respect to the output value
   *  (if you prefer computing it wrt the net value you can do so but it requires more computing power.
   *  @param  x The value by which the gradient is determined.
   *  @return double  the gradient at x.
   */
  public double outputFunctionDerivative(double x) {
    return x*(1.0-x);
  }

  /** Computes the output values of the output nodes in the network 
   * 	given input values.
   *  @param  x  The input values.
   *  @return double[]    The vector of computed output values
   */
  public double[] feedforward(double[] x) {
	  emptyArrays();
	  
	  for (int i = 0; i < in; i++) {
		  for (int j = 0; j < h1; j++) {
			  hidden1[j] += input[i] * w[j][i] + bias[j];
		  }
	  }
	  
	  for (int i = 0; i < h1; i++) {
		  for (int j = 0; j < h2; j++) {
			  hidden2[j] += hidden1[i] * w2[j][i] + bias2[j];
		  }
	  }
	  
	  for (int i = 0; i < h2; i++) {
		  for (int j = 0; j < out; j++) {
			  output[j] += outputFunction(hidden2[i]) * w3[j][i];
		  }
	  }
	  
	  for (int i = 0; i < out; i++) {
		  output[i] = outputFunction(output[i] + bias3[i]);
	  }
	  
	  return output;
  }
  
  /**
   * Empty each layer.
   */
  private void emptyArrays() {
	  for (int i = 0; i < h1; i++) hidden1[i] = 0;
	  
	  for (int i = 0; i < h2; i++) hidden2[i] = 0;
	  
	  for (int i = 0; i < out; i++) output[i] = 0;
  }
  
  

  /** Adapts weights in the network given the specification of which values that should appear at the output (target)
   *  when the input has been presented.
   *  The procedure is known as error backpropagation. This implementation is "online" rather than "batched", that is,
   *  the change is not based on the gradient of the global error, merely the local -- pattern-specific -- error.
   *  @param  x  The input values.
   *  @param  d  The desired output values.
   *  @param  eta     The learning rate, always between 0 and 1, typically a small value, e.g. 0.1
   *  @return double  An error value (the root-mean-squared-error).
   */
  public double train(double[] x, double[] d, double eta) {
	double err = 0;
	clearErrors();
	input = x.clone();
	
	// present the input and calculate the outputs
	feedforward(x);
	
	
	for (int k = 0; k < out; k++) {
		error[k] =  (d[k] - output[k]) * outputFunctionDerivative(output[k]);
	}
	
	for (int k = 0; k < h2; k++) {
		error3[k] = 0;
		for (int n = 0; n < out; n++) {
			error3[k] += error[n] * w3[n][k];
		}
	}
	
	for (int k = 0; k < h1; k++) {
		error2[k] = 0;
		for (int n = 0; n < h2; n++) {
			error2[k] += error3[n] * w2[n][k];
		}
	}
	
	for (int p = 0; p < h2; p++) {
		error3[p] = error3[p] * outputFunctionDerivative(hidden2[p]);
	}
	for (int p = 0; p < h1; p++) {
		error2[p] = error2[p] * outputFunctionDerivative(hidden1[p]);
	}
	
	for (int j = 0; j < out; j++) {
		  for (int i = 0; i < h2; i++) {
			  w3[j][i] = eta * error[j] * hidden2[i];
		  }
	}
	
	for (int j = 0; j < h2; j++) {
		  for (int i = 0; i < h1; i++) {
			  w2[j][i] = eta * error3[j] * hidden1[i];
		  }
	}
	
	for (int j = 0; j < h1; j++) {
		  for (int i = 0; i < in; i++) {
			  w[j][i] = eta * error2[j] * input[i];
		  }
	}
	  
	// Calculate overall error
	for (int i = 0; i < out; i++) {
		double diff = output[i] - d[i];
		err += diff * diff;
	}
	
	return Math.sqrt(err/out);
  }

  /**
   * Set the error arrays to 0. 
   */
  private void clearErrors() {
	  for (int i = 0; i < h1; i++) {
		  error2[i] = 0;
	  }
	  
	  for (int i = 0; i < h2; i++) {
		  error3[i] = 0;
	  }
	  
	  for (int i = 0; i < out; i++) {
		  error[i] = 0;
	  }
	  return;
  }



}