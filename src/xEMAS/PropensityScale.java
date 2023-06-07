/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    PropensityScale.java
 *    Copyright (C) 2009 Yasser EL-Manzalawy
 *
 */
package epit.classifiers.propensity;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.core.RevisionUtils;
import weka.core.Capabilities.Capability;
import java.util.Enumeration;
import java.util.Vector;
import java.util.StringTokenizer;

/**
 * <!-- globalinfo-start --> A classifier for assigning propensity scores
 * to sequence windows using a specified scale (e.g., Parker's hydrophilicity scale).<br>
 *
 * <p>References: <br>
 * Parker, J. and Guo, D and, H. R. (1986). New hydrophilicity scale derived from high-performance liquid chromatography peptide retention data: correlation of predicted surface residues with antigenicity and x-ray-derived accessible sites. Biochemistry, 25:5425-5432.<br>
 * <p/> <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are: <p/>
 *
 * <pre>
 * -L &lt;int&gt;
 *  The window size (default -1)
 * </pre>
 *
 * <pre>
 * -A &lt;string&gt;
 *  The sequence alphabet (default standard 20 amino acids).
 * </pre>
 * <pre>
 * -S &lt;string&gt;
 *  The propenisity scale index (default = Parker's hydrophilicity scale).
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Yasser EL-Manzalawy (yasser@cs.iastate.edu)
 * @version $Revision: 0.9 $
 */
public class PropensityScale extends Classifier {

    /** for serialization */
    static final long serialVersionUID = -7378107808933117974L;

    // should be comma seprated and no spaces are allowed
    private String parker = "2.1,4.2,7.0,10.0,1.4,6.0,7.8,5.7,2.1,-8.0,-9.2,5.7,-4.2,-9.2,2.1,6.5,5.2,-10.0,-1.9,-3.7";
    private double[] m_Scale;
    /** Amino acid index. Should be entered as a single string (e.g., no spaces are allowed).
    Values are comma separated. Default value is Parker's hydrophilicity scale. */
    protected String m_ScaleStr = parker;
    /** The sequence alphabet (default = standards 20 amino acids). */
    protected String m_AA = "ARNDCQEGHILKMFPSTWYV";
    /** Size of the window. Default is -1 which means that no specific window size
    is used during the training and therefore the learned model can be used to assign
    scores to sequence windows of any arbitrary length. */
    protected int m_Size = -1;

    /**
     * Returns a string describing classifier
     *
     * @return a description suitable for displaying in the
     *         explorer/experimenter gui
     */
    public String globalInfo() {
        return "Class for assigning antigenicity scores using a specified propenisity scale.";
    }

    /**
     * Sets the alphabet parameter tip text in the Weka GUI.
     * @return tip text.
     */
    public String alphabetTipText() {
        return "The squence alphabet.";
    }

    /**
     * Gets the sequence alphabet. 
     * @return The sequence alphabet.
     */
    public String getAlphabet() {
        return m_AA;
    }

    /**
     * Sets the sequence alphabet.
     * @param alphabet The sequence alphabet.
     */
    public void setAlphabet(String aa) {
        m_AA = aa;
    }

    /**
     * Sets the scale parameter tip text in the Weka GUI.
     * @return tip text.
     */
    public String scaleTipText() {
        return "Propenisity Scale.";
    }

    /**
     * Sets the propensity scale.
     * @param scale The propensity scale.
     */
    public void setScale(String scale) {
        m_ScaleStr = scale;
    }

    /**
     * Gets the propensity scale.
     * @return propensity scale.
     */
    public String getScale() {
        return m_ScaleStr;
    }

    /**
     * Sets window size parameter tip text in the Weka GUI.
     * @return tip text.
     */
    public String sizeTipText() {
        return "Window size.";
    }

    /**
     * Gets the window size.
     * @return window size.
     */
    public int getSize() {
        return m_Size;
    }

    /**
     * Sets the window size.
     * @param size The window size.
     */
    public void setSize(int size) {
        m_Size = size;
    }

    /**
     * Lists valid options for that classifier.
     * <!-- options-start --> Valid options are: <p/>
     * <pre>
     * -L &lt;int&gt;
     *  The window size  (default -1)
     * </pre>
     *
     * <pre>
     * -A &lt;string&gt;
     *  The sequence alphabet (default standard 20 amino acids).
     * </pre>
     * <pre>
     * -S &lt;string&gt;
     *  The propenisity scale index (default = Parker's hydrophilicity scale).
     * </pre>
     * <!-- options-end -->
     *
     */
    public Enumeration listOptions() {

        Vector newVector = new Vector();

        newVector.addElement(new Option("\t Window size\n", "L", 1,
                "-L <int>"));

        newVector.addElement(new Option("\t Sequence alphabet\n", "A", 1,
                "-A <string>"));
        newVector.addElement(new Option("\t Propensity scale\n", "S", 1,
                "-S <string>"));

        return newVector.elements();
    }

    /**
     * Parses a given list of options. 
     *
     *
     * @param options
     *            the list of options as an array of strings
     * @throws Exception
     *             if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        String tmpStr = Utils.getOption('L', options);
        if (tmpStr.length() != 0) {
            setSize(Integer.parseInt(tmpStr));
        } else {
            setSize(-1);
        }

        tmpStr = Utils.getOption('A', options);
        if (tmpStr.length() != 0) {
            setAlphabet(tmpStr);
        } else {
            setAlphabet("ACDEFGHIKLMNPQRSTVWY");
        }

        tmpStr = Utils.getOption('S', options);
        if (tmpStr.length() != 0) {
            setScale(tmpStr);
        } else {
            setScale(parker);
        }

        super.setOptions(options);
    }

    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {
        int size = 6;
        if (getScale().length() == 0) {
            size = 4;
        }
        String[] options = new String[size];
        int current = 0;
        options[current++] = "-L";
        options[current++] = "" + getSize();
        options[current++] = "-A";
        options[current++] = "" + getAlphabet();
        if (getScale().length() != 0) {
            options[current++] = "-S";
            options[current++] = getScale();
        }
        return options;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();

        // attributes
        // TODO: only string attributes
        result.enableAllAttributes();
        result.enable(Capability.MISSING_VALUES);

        // class
        //TODO: enable only binary and numeric classes
        //result.enableAllClasses();
        result.enable(Capability.BINARY_CLASS);
        result.enable(Capability.NUMERIC_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        result.enable(Capability.NO_CLASS);

        return result;
    }

    /**
     * Train a classifier.
     *
     * @param data
     *            the training data.
     * @throws Exception
     *             if the classifier could not be built successfully
     */
    public void buildClassifier(Instances data) throws Exception {
        if (!data.attribute(0).isString() || data.numAttributes() != 2) {
            throw new Exception(
                    "Illegal input format. Should be <string, class>.");
        }

        StringTokenizer st = new StringTokenizer(m_ScaleStr, ", \t\n", false);
        int size = st.countTokens();
        if (size != m_AA.length()) {
            throw new Exception(
                    "Size of the propensity scale does not match the alphabet size.");
        }
        m_Scale = new double[size];
        for (int i = 0; i < size; i++) {
            m_Scale[i] = Double.parseDouble(st.nextToken());
        }
        if (m_Size != -1) {
            if (m_Size % 2 == 0) {
                throw new Exception("Window size should be an odd number");
            }
            if (data.instance(0).stringValue(0).length() != m_Size) {
                throw new Exception("Window size and length of the instance sequence are not equal");
            }
        }    
    }

    /**
     * Calculates the class membership probabilities for the given test
     * instance.
     *
     * @param instance
     *            the instance to be classified
     * @return predicted class probability distribution
     * @throws Exception
     *             if instance could not be classified successfully
     */
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] distribution = new double[2];
        String window = instance.stringValue(0);
        distribution[1] = 1 / (1 + Math.exp(-scoreWindow(window)));
        distribution[0] = 1 - distribution[1];

        return distribution;
    }

    /**
     * Calculates the predicted numerical label (for regression tasks).
     * @param instance input instance
     * @return
     * @throws java.lang.Exception
     */
    public double classifyInstance(Instance instance) throws Exception {
        String window = instance.stringValue(0);
        return 1 / (1 + Math.exp(-scoreWindow(window)));
    }

    /**
     * Determines the antigenicity score of an input sequence window.
     * @param window
     * @return
     * @throws java.lang.Exception
     */
    private double scoreWindow(String window) throws Exception {
        double len = window.length();
        if (m_Size != -1) {
            if (m_Size != len) {
                throw new Exception("Length of test instance and training window size do not match");
            }
        }
        double sum = 0.0;
        int index;
        for (int i = 0; i < len; i++) {
            index = m_AA.indexOf(window.substring(i, i + 1));
            if (index == -1) {
                System.err.print("\n Illegal alphabet symbol: " + window.substring(i, i + 1) + " will be skipped.");
            } else {
                sum += m_Scale[index];
            }
        }
        sum /= len;
        return sum;
    }

    /**
     * Returns description of the boosted classifier.
     *
     * @return description of the boosted classifier as a string
     */
    public String toString() {
        String text = "Propenisity Scale based classifier by Yasser EL-Manzalawy \n"
                + "Amino acids " + m_AA + "\n"
                + " Scale " + m_ScaleStr
                + "\n";
        return text;
    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 0.9 $");
    }

    /**
     * Main method for testing this class.
     *
     * @param argv
     *            the options
     */
    public static void main(String[] argv) {
        runClassifier(new PropensityScale(), argv);
    }
}
