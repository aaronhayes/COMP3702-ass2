package machl;
import java.util.*;
import java.io.IOException;
import java.io.Serializable;

import bitmap.ClassifiedBitmap;
import bitmap.LetterClassifier;

/**
 * A implementation of a multi-layered neural network.
 * 	Adapted from the single-layer network provided (NN1.java)
 */


public class MLNN implements Serializable {
  public double[] y;      // the values produced by each node (indices important, see weights/biases)
  public double[] h1;
  public double[] h2;
  
  public double[][] w;    // the trainable weight values [to node][from input]
  public double[][] wH1;
  public double[][] wH2;
  
  public double[] bias;   // the trainable bias values for nodes
  public double[] biasH1;
  public double[] biasH2;
  
  public double[] output_error;
  public double[] h1_error;
  public double[] h2_error;
  public int nInput;      // Number of input Nodes
  
  public double out_change;
  public double h1_change;
  public double h2_change;
  
  Random rand;            // a random number generator for initial weight values

  /** Constructs a neural network structure and initializes weights to small random values.
   *  @param  nInput  Number of input nodes
   *  @param  nOutput Number of output nodes
   *  @param  nH1     Number of hidden layer 1 nodes
   *  @param  nH2     Number of hidden layer 2 nodes
   *  @param  seed    Seed for the random number generator used for initial weights.
   *
   */
  public MLNN(int nInput, int nOutput, int nH1, int nH2, int seed) {
	// allocate space for node and weight values
    y = new double[nOutput];
    h1 = new double[nH1];
    h2 = new double[nH2];
    
    this.nInput = nInput;
    
    w = new double[nOutput][nH2];
    wH1 = new double[nH1][nInput];
    wH2 = new double[nH2][nH1];
    
    bias = new double[nOutput];
    biasH1 = new double[nH1];
    biasH2 = new double[nH2];
	  
    initialiseValues(seed);
    
    }
  
  /**
   * Initialize weight and bias values
   */
  public void initialiseValues(int seed) {
	  rand = new Random(seed);
	  for (int j = 0; j < y.length; j++) {
		  for (int i = 0; i < h2.length; i++) {
			  w[j][i] = rand.nextGaussian() * .1;
		  }
		  bias[j] = rand.nextGaussian() * 0.1;
	  }
	  
	  for (int j = 0; j < h2.length; j++) {
		  for (int i = 0; i < h1.length; i++) {
			  wH2[j][i] = rand.nextGaussian() * 0.1;
		  }
		  biasH2[j] = rand.nextGaussian() * 0.1;
	  }
	  
	  for (int j = 0; j < h1.length; j++) {
		  for (int i = 0; i < nInput; i++) {
			  wH1[j][i] = rand.nextGaussian() * 0.1;
		  }
		  biasH1[j] = rand.nextGaussian() * 0.1;
	  }
  }
  /** The so-called output function. Computes the output value of a node given the summed incoming activation.
   *  You can use anyone you like if it is differentiable.
   *  This one is called the logistic function (a sigmoid) and produces values bounded between 0 and 1.
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

  /** Computes the output values of the output nodes in the network given input values.
   *  @param  x  The input values.
   *  @return double[]    The vector of computed output values
   */
  public double[] feedforward(double[] x) {
	  
	  for (int j = 0; j < h1.length; j++) h1[j] = 0;
	  for (int j = 0; j < h2.length; j++) h2[j] = 0;
	  for (int j = 0; j < y.length; j++) y[j] = 0;
	  
	  // Feed to Hidden Layer 1
	  for (int i = 0; i < x.length; i++) {
		  for (int j = 0; j < h1.length; j++) {
			  h1[j] += x[i] * wH1[j][i];
		  }
	  }
	  // Add Bias To Hidden Layer 1
	  for (int k = 0; k < h1.length; k++) {
		  h1[k] = outputFunction(h1[k] + biasH1[k]);
	  }
	  
	  // Feed to Hidden Layer 2
	  for (int i = 0; i < h1.length; i++) {
		  for (int j = 0; j < h2.length; j++) {
			  h2[j] += h1[i] * wH2[j][i];
		  }
	  }
	  // Add Bias To Hidden Layer 2
	  for (int k = 0; k < h2.length; k++) {
		  h2[k] = outputFunction(h2[k] + biasH2[k]);
	  }
	  
	  
	  // Feed to output Layer
	  for (int i = 0; i < h2.length; i++) {
		  for (int j = 0; j < y.length; j++) {
			  y[j] += h2[i] * w[j][i];
		  }
	  }
	  
	  for (int k = 0; k < y.length; k++) {
		  y[k] = outputFunction(y[k] + bias[k]);
	  }
	  
    return y;
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

    // present the input and calculate the outputs
    feedforward(x);

    // allocate space for errors of individual nodes
    output_error = new double[y.length];
    h1_error = new double[h1.length];
    h2_error = new double[h2.length];
    
    // compute the error of output nodes (explicit target is available -- so quite simple)
    // also, calculate the root-mean-squared-error to indicate progress
    double rmse = 0;
    for (int j = 0; j < y.length; j++) {
      double diff = d[j] - y[j];
      output_error[j] = diff * outputFunctionDerivative(y[j]);
      rmse += diff * diff;
    }
    
    // Calculate the error of the hidden nodes
    for (int k = 0; k < h2.length; k++) {
    	h2_error[k] = 0;
    	for (int n = 0; n < y.length; n++) {
    		h2_error[k] += output_error[n] * w[n][k];
    	}
    	h2_error[k] *= outputFunctionDerivative(h2[k]);
    }   
    
    for (int k = 0; k < h1.length; k++) {
    	h1_error[k] = 0;
    	for (int n = 0; n < h2.length; n++) {
    		h1_error[k] += h2_error[n] * wH2[n][k];	
    	}
    	h1_error[k] *= outputFunctionDerivative(h1[k]);
    }
    

    
    rmse = Math.sqrt(rmse/y.length);

    // change weights according to errors
    for (int j = 0; j < y.length; j++) {
      for (int i = 0; i < h2.length; i++) {
        w[j][i] += output_error[j] * h2[i] * eta;
      }
      bias[j] += output_error[j] * 1.0 * eta; // bias can be understood as a weight from a node which is always 1.0.
    }
    
    for (int j = 0; j < h2.length; j++) {
        for (int i = 0; i < h1.length; i++) {
          wH2[j][i] += h2_error[j] * h1[i] * eta;
        }
        biasH2[j] += h2_error[j] * 1.0 * eta; 
      }
    
    for (int j = 0; j < h1.length; j++) {
        for (int i = 0; i < x.length; i++) {
        	wH1[j][i] += h1_error[j] * x[i] * eta;
        }
        biasH1[j] += h1_error[j] * 1.0 * eta;
    }

    
    return rmse;
  }


}