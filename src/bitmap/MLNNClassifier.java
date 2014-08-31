package bitmap;

import java.io.IOException;
import java.util.Random;

import javax.management.MXBean;

import machl.*;

/**
 * A neural network handwritten letter recognizer. 
 * 
 */

public class MLNNClassifier extends LetterClassifier {

  private static String name = "Advanced Classifer";
  private MLNN[] nn = null;
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
		double[] res = test(map, x);
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
  private double[] test(Bitmap map, int x) {
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
		    	if (p % 5000 == 0 && isTrainingComplete(x) == true) break classX;
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
	            double[] res = test((Bitmap)bitmaps[i], x);
	            int actual = 0;
	            for (int k = 0; k < res.length; k++) {
	                if (res[k] > res[actual])
	                  actual = k;
	              }
	            int target = bitmaps[i].getTarget();
	            if (target == actual) correct++;
	        }
	        // If accuracy is greater than 80% training is complete
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
  public MLNNClassifier(int nRows, int nCols, int h1, int h2) {
    rand=new Random(System.currentTimeMillis());
    
    int in = nRows * nCols; //System.out.println("num in : " + in);
    int out = getClassCount(); //System.out.println("num out : " + out);
    nn = new MLNN[1]; // Number of classifiers in ensemble
    
    for (int x = 0; x < nn.length; x++) {
    	nn[x] = new MLNN(in, out, h2, h1, rand.nextInt());
    }
    
    targets=new double[getClassCount()][getClassCount()];
    for (int c=0; c<getClassCount(); c++)
      targets[c][c]=1;
  }

}