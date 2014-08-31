package bitmap;

import java.io.IOException;
import java.io.Serializable;
import java.util.Random;

import javax.management.MXBean;

import machl.*;

/**
 * A neural network handwritten letter recognizer. 
 * 
 */

public class Classifier_42338468 extends LetterClassifier {

  private static String name = "Classifier 42338468";
  private NN_42338468[] nn = null;
  private Random rand;
  private double[][] targets = null; // target vectors;
  
  /**
   * Identifies the classifier, e.g. by the name of the author/contender, or by whatever you want to
   * identify this instance when loaded elsewhere.
   * @return the identifier
   */
  public String getName() {
    return name;
  }

  /**
   * Classifies the bitmap
   * @param map the bitmap to classify
   * @return the probabilities of all the classes 
   * 	(should add up to 1 * nn.length).
   */
  public double[] test(Bitmap map) {
	double[] out = new double[getClassCount()];
	for (int x = 0; x < nn.length; x++) {
		double[] res = runTest(map, x);
		for (int o = 0; o < res.length; o++) {
			out[o] += res[o];
		}
	}
    return out;
  }
  
  /**
   * Classifies the bitmap for a single classifier within an ensemble
   * @param map the bitmap to classify
   * @return the probabilities of all the classes should add up to 1.
   */
  private double[] runTest(Bitmap map, int x) {
		return nn[x].feedforward(process(map).toDoubleArray());
	}
  
  /**
   * Trains the neural network classifier on randomly picked samples from specified training data.
   * @param maps the bitmaps which are used as training inputs including targets
   * @param nPresentations the number of samples to present
   * @param eta the learning rate
   */
  public void train(ClassifiedBitmap[] maps, int nPresentations, double eta) {	  
	  
	  for (int x = 0; x < nn.length; x++) {
		  classX : for (int p=0; p<nPresentations; p++) {
			  int sample = rand.nextInt(maps.length); 
		    	Bitmap preMap = process((Bitmap)maps[sample]);
		    	nn[x].train(preMap.toDoubleArray(), 
		    			targets[maps[sample].getTarget()], eta+(eta*x/2));
		    	
		    	/* If network has reached over training, break */
		    	//if (p % 5000 == 0 && isTrainingComplete(x) == true) break classX;
		  }
	  }
	    return;	    
  }
  
  /**
   * Checks if the network is being over trained and training 
   * 	should be completed.
   * @param x index of the classifier in the ensemble
   * @return boolean to indicate if training is complete
   */
  private boolean isTrainingComplete(int x) {
	  try {
	        ClassifiedBitmap[] bitmaps = LetterClassifier.loadLetters("verificationData.txt");

	        int correct = 0;
	        for (int i=0; i<bitmaps.length; i++) {
	            double[] res = runTest((Bitmap)bitmaps[i], x);
	            int actual = 0;
	            for (int k = 0; k < res.length; k++) {
	                if (res[k] > res[actual])
	                  actual = k;
	              }
	            int target = bitmaps[i].getTarget();
	            if (target == actual) correct++;
	        }
	        // If accuracy is greater than 90% training is complete
	        double ratio = ((double) correct)/bitmaps.length;
	        if (ratio >= 0.8) return true;
	        
	  } catch (IOException ex) {
		  	// Cannot find verification data
	        return false;
	  }
	  
	  return false;
  }
  


/**
   * Count the number of surrounding pixels that are set
   * @param map The Image
   * @param row the row of the pixel
   * @param col the column of the pixel
   * @return count the number of pixels that are set around the given pixel
   */
  private int getSurrounding(Bitmap map, int row, int col) {
	int count = 0;
	
	for (int i = row - 1; i <= row + 1; i++) {
		for (int j = col - 1; j <= col + 1; j++) {
			if (map.get(i, j)) count++;
		}
	}
	return count;
  }
  
  /**
   * Pre-processing on the bitmap image
   * @param bitmap The bitmap to be processed
   * @return processed bitmap
   */
  private Bitmap process(Bitmap oldMap) {
	Bitmap map = new Bitmap(32, 32);
	double removed = 0;
	double set = 0;
	
	/* Remove Noise */
	for (int i = 0; i < oldMap.getCols(); i++) {
		for(int j = 0; j < oldMap.getRows(); j++) {
			if (oldMap.get(j, i)) set++;
			int num = getSurrounding(oldMap, j, i);
			if (num < 4 && oldMap.get(j, i)) {
				map.set(j, i, false);
				removed++;
			} else {
				map.set(j, i, oldMap.get(j, i));
			}
		}
	}
	
	/* Check if too much information has been removed */
	double ratio = removed/set;
	if (ratio >= 0.25) {
		for (int i = 0; i < oldMap.getCols(); i++) {
			for(int j = 0; j < oldMap.getRows(); j++) {
				map.set(j, i, oldMap.get(j, i));
			}
		}
	}
	
	/* Fill Gaps */
	for (int i = 0; i < oldMap.getCols(); i++) {
		for(int j = 0; j < oldMap.getRows(); j++) {
			int num = getSurrounding(oldMap, j, i);
			if (num > 4) {
				map.set(j, i, true);
			} 
		}
	}
	
	return map;
}

/**
   * Construct a neural network classifier for bitmaps of specified size.
   * @param nRows number of rows in the bitmap
   * @param nCols number of columns in the bitmap
   */
  public Classifier_42338468(int nRows, int nCols, int h1, int h2) {
    rand=new Random(System.currentTimeMillis());
    
    int in = nRows * nCols; //System.out.println("num in : " + in);
    int out = getClassCount(); //System.out.println("num out : " + out);
    nn = new NN_42338468[3]; // Number of classifiers in ensemble
    
    for (int x = 0; x < nn.length; x++) {
    	nn[x] = new NN_42338468(in, out, h2, h1, rand.nextInt());
    }
    
    targets=new double[getClassCount()][getClassCount()];
    for (int c=0; c<getClassCount(); c++)
      targets[c][c]=1;
  }

  
  
  /**
   * Multi Layer Neural Network
   */
private class NN_42338468 implements Serializable {
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
	  
	  Random rand;  // a random number generator for initial weight values

	  /** Constructs a neural network structure and initializes weights to small random values.
	   *  @param  nInput  Number of input nodes
	   *  @param  nOutput Number of output nodes
	   *  @param  nH1     Number of hidden layer 1 nodes
	   *  @param  nH2     Number of hidden layer 2 nodes
	   *  @param  seed    Seed for the random number generator used for initial weights.
	   *
	   */
	  public NN_42338468(int nInput, int nOutput, int nH1, int nH2, int seed) {
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
}