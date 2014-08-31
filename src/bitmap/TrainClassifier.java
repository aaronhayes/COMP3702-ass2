package bitmap;

import java.io.*;

/**
 * This program trains a classifier and saves it in a file to be read when used.
 * @author Mikael Boden
 * @version 1.0
 */

public class TrainClassifier {

  public TrainClassifier(String[] args) {
    // create the classifier
    //NNClassifier c=new NNClassifier(32, 32);
	
	// Multi-layer neural network
	// MLNNClassifier c = new MLNNClassifier(32, 32, 100, 75);
    
	// Competiton classifier
	Classifier_42338468 c = new Classifier_42338468(32,32,100,75);
	
	// or - for ID3 - replace with the following line of code
    //ID3Classifier c=new ID3Classifier(32, 32);

    // load data
    try {
      ClassifiedBitmap[] bitmaps=LetterClassifier.loadLetters(args[1]);
      // train it using all available training data
      double eta = 0.2;
      c.train(bitmaps, 25000, eta);
      // or - for ID3 - replace with the following line of code
      //c.train(bitmaps,0.8,5);
    } catch (IOException ex) {
      System.err.println("Error loading data.txt: "+ex.getMessage());
    }
    try {
      Classifier.save(c, args[0]);
    } catch (Exception ex) {
      System.err.println("Failed to serialize and save file: "+ex.getMessage());
    }
  }

  public static void main(String[] args) {
    if (args.length!=2) {
      System.err.println("Usage: TrainClassifier <classifier-file> <bitmap-file>");
      System.exit(1);
    }
    long startTime = System.nanoTime();
    new TrainClassifier(args);
    long endTime = System.nanoTime();
    double time = (endTime - startTime) / 1000000000.0;
    System.out.println("");
    System.out.println("Classifier Created in " + time + " seconds.");
    System.out.println("Done.");
  }

}