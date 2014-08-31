package bitmap;

import java.io.*;
import java.util.*;

/**
 * <p>This class extends Classifier and provides some functionality specific to letter recognition.</p>
 * @author Mikael Boden
 * @version 1.0
 */

public class LetterClassifier extends Classifier {

  private static String name="Letter Classifier";
  private static int nClasses=('Z'-'A')+1+9;

  /**
   * Identifies the classifier, e.g. by the name of the author/contender
   * @return the identifier
   */
  public String getName() {
    return name;
  }

  /**
   * Determines the number of possible classes that this classifier discriminates between.
   * @return the number of classes
   */
  public static int getClassCount() {
    return nClasses;
  }

  /**
   * Determine the name of the class specified by index (0..getClassCount)
   * @param index the index number of the class
   * @return the label/name of the specified class
   */
  public String getLabel(int index) {
	  Character letter;
	  if (index<('Z'-'A'+1)){
		  letter=new Character((char)('A'+index));
	  } else {
		  letter=new Character((char)('1'+(index-('Z'-'A'+1))));
	  }
    return letter.toString();
  }

  /**
   * Helper method for loading a text-file with classified bitmaps (each representing a letter).
   * It ignores rows with format problems.
   * @param filename the filename of the text-file that holds the classified bitmaps
   * @return an array of classified bitmaps
   * @throws IOException if the file operation fails
   */
  public static ClassifiedBitmap[] loadLetters(String filename) throws IOException {
    Vector bmaps=new Vector();
    BufferedReader reader=new BufferedReader(new FileReader(filename));
    String line=reader.readLine();
    while (line!=null) {
      ClassifiedBitmap bmap=null;
      try {
        bmap=new ClassifiedBitmap(line);
        bmaps.add(bmap);
      } catch (RuntimeException ex) {
        ; // the line does not conform to the Bitmap format or does not specify the target class correctly
      }
      line=reader.readLine();
    }
    ClassifiedBitmap[] bitmaps=new ClassifiedBitmap[bmaps.size()];
    Iterator iter=bmaps.iterator();
    for (int i=0; iter.hasNext(); i++)
      bitmaps[i]=(ClassifiedBitmap)iter.next();
    return bitmaps;
  }

}