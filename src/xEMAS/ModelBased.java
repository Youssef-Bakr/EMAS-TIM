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
 *    ModelBased.java
 *    Copyright (C) 2009 Yasser EL-Manzalawy
 *
 */
package epit.classifiers.meta;

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
import java.io.ObjectInputStream;
import java.io.FileInputStream;

/**
 * <!-- globalinfo-start --> A meta classifier for performing classification/regression
 * using a specified model file. This meta classifier allows users to get consensus predictions
 * over a test set using several existing model files. For mor information see EpiT tutorial.
 *
 * <p/> <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are: <p/>
 *
 * <pre>
 * -M &lt;string&gt;
 *  The Model file.
 * </pre>
 *
 *
 * <!-- options-end -->
 *
 * @author Yasser EL-Manzalawy (yasser@cs.iastate.edu)
 * @version $Revision: 0.9 $
 */
public class ModelBased extends Classifier {

    /** for serialization */
    static final long serialVersionUID = -7378107808933117974L;
    /** Full path and name of the weka model file which will be used to
     * classify test instances.
     */
    protected String m_ModelFile = "";
    private Classifier m_Model = null;

    /**
     * Returns a string describing classifier
     *
     * @return a description suitable for displaying in the
     *         explorer/experimenter gui
     */
    public String globalInfo() {
        return "Class for using specified model file for classifying test instances.\n";
    }

    /**
     * Sets the model file parameter tip text in the Weka GUI.
     * @return tip text.
     */
    public String modelFileTipText() {
        return "The model file.";
    }

    /**
     * Gets the model file name.
     * @return The sequence alphabet.
     */
    public String getModelFile() {
        return m_ModelFile;
    }

    /**
     * Sets the model file name.
     * @param file_name the model file name.
     */
    public void setModelFile(String file_name) {
        m_ModelFile = file_name;
    }

    /**
     * Lists valid options for that classifier.
     * <!-- options-start --> Valid options are: <p/>
     * <pre>
     * -M &lt;int&gt;
     *  The model file name.
     * </pre>
     *
     * <!-- options-end -->
     *
     */
    public Enumeration listOptions() {

        Vector newVector = new Vector();

        newVector.addElement(new Option("\t Model file name\n", "L", 1,
                "-M <string>"));

        return newVector.elements();
    }

    /**
     * Parses a given list of options.
     * <!-- options-start --> Valid options are: <p/>
     *
     * <pre>
     * -M &lt;string&gt;
     *  The Model file.
     * </pre>
     *
     *
     * <!-- options-end -->
     *
     * @param options
     *            the list of options as an array of strings
     * @throws Exception
     *             if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        String tmpStr = Utils.getOption('M', options);
        if (tmpStr.length() != 0) {
            setModelFile(tmpStr);
        } else {
            setModelFile("");
        }

        super.setOptions(options);
    }

    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {
        String[] options = new String[2];
        int current = 0;
        if (getModelFile().length() > 0) {
            options[current++] = "-M";
            options[current++] = getModelFile();
        }

        while (current < options.length) {
            options[current++] = "";
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

        result.enableAllAttributes();
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enableAllClasses();
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
     *             if the model can not be loaded from model file.
     */
    public void buildClassifier(Instances data) throws Exception {
        // loads the model
        ObjectInputStream model = new ObjectInputStream(new FileInputStream(getModelFile()));
        m_Model = (Classifier) model.readObject();
        model.close();
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
        return m_Model.distributionForInstance(instance);
    }

    /**
     * Calculates the predicted numerical label (for regression tasks).
     * @param instance input instance
     * @return
     * @throws java.lang.Exception
     */
    public double classifyInstance(Instance instance) throws Exception {
        return m_Model.classifyInstance(instance);
    }

    /**
     * Returns description of the boosted classifier.
     *
     * @return description of the boosted classifier as a string
     */
    public String toString() {
        String text = "Model based classifier by Yasser EL-Manzalawy \n";
        if (m_Model != null){
            text += "Base model \n"
                + m_Model.toString()
                + "\n";
        }
                
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
     *            The options.
     */
    public static void main(String[] argv) {
        runClassifier(new ModelBased(), argv);
    }
}
