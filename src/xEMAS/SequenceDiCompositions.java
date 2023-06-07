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
 *    SequenceDiCompositions.java
 *    Copyright (C) 2009 Yasser EL-Manzalawy
 *
 */
package epit.filters.unsupervised.attribute;

import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.UnsupervisedFilter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;

/**
 * <!-- globalinfo-start --> Produces the sequence dipeptides composition features of input sequences
 * (e.g., the amino acid dipeptide composition features of protein sequences). <p/> <!-- globalinfo-end
 * -->
 *
 * <!-- options-start --> Valid options are: <p/>
 *
 * <pre>
 * -A &lt;string&gt;
 *  The alphabet (default standard 20 amino acids)
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Yasser EL-Manzalawy (yasser@cs.iastate.edu)
 * @version $Revision: 0.9 $
 */
public class SequenceDiCompositions extends Filter implements
        UnsupervisedFilter, OptionHandler {

    /** for serialization */
    static final long serialVersionUID = 3119607037607101160L;
    /** The sequence alphabet (default = standard 20 amino acids). */
    protected String m_AA = "ACDEFGHIKLMNPQRSTVWY";

    /**
     * Returns a string describing this classifier
     *
     * @return a description of the classifier suitable for displaying in the
     *         explorer/experimenter gui
     */
    public String globalInfo() {
        return "SequenceDiCompositions filter by Yasser EL-Manzalawy.";
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
     * Lists valid options for that filter.
     * <!-- options-start --> Valid options are: <p/>
     *
     * <pre>
     * -A &lt;string&gt;
     *  The alphabet (default standard 20 amino acids)
     * </pre>
     *
     *
     * <!-- options-end -->
     *
     */
    public Enumeration listOptions() {

        Vector newVector = new Vector(1);
        newVector.addElement(new Option("\tSets sequence alphabet\n", "A", 1,
                "-A <string>"));

        return newVector.elements();
    }

    /**
     * Sets filter options.
     * -A sequence alphabet (default 20 amino acids).
     * -S scaling method  (default none).
     */
    public void setOptions(String[] options) throws Exception {

        String tmpStr = Utils.getOption('A', options);
        if (tmpStr.length() != 0) {
            setAlphabet(tmpStr);
        } else {
            setAlphabet("ACDEFGHIKLMNPQRSTVWYX");
        }

    }

    /**
     * Returns current values for the filter options.
     */
    public String[] getOptions() {

        String[] options = new String[2];
        int current = 0;
        options[current++] = "-A";
        options[current++] = "" + getAlphabet();
        return options;
    }

    /**
     * Returns the Capabilities of this filter.
     *
     * @return the capabilities of this object
     * @see Capabilities
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();

        // attributes //TODO: only string attributes
        result.enableAllAttributes();
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enableAllClasses();
        result.enable(Capability.MISSING_CLASS_VALUES);
        result.enable(Capability.NO_CLASS);

        return result;
    }

    /**
     * Sets the format of the input instances.
     *
     * @param instanceInfo
     *            an Instances object containing the input instance structure
     *            (any instances contained in the object are ignored - only the
     *            structure is required).
     * @return true if the outputFormat may be collected immediately
     * @throws Exception
     *             if the input format can't be set successfully
     */
    public boolean setInputFormat(Instances instanceInfo) throws Exception {

        super.setInputFormat(instanceInfo);
        m_FirstBatchDone = false;

        if (!instanceInfo.attribute(0).isString() || instanceInfo.numAttributes() != 2) {
            throw new Exception("Illegal input format. Should be <string, class>.");
        }

        int len = m_AA.length() * m_AA.length();
        FastVector attInfo = new FastVector(len + 1);
        for (int i = 0; i < len; i++) {
            attInfo.addElement(new Attribute("a" + i));
        }
        Attribute labelAttr = (Attribute) getInputFormat().classAttribute().copy();
        attInfo.addElement(labelAttr);
        Instances outputFormat = new Instances(instanceInfo.relationName() + "-" + toString(), attInfo, 0);
        outputFormat.setClassIndex(outputFormat.numAttributes() - 1);
        setOutputFormat(outputFormat);

        return true;
    }

    /**
     * Input an instance for filtering.
     * @param instance  the input instance
     * @return true if the filtered instance may now be
     * collected with output().
     * @throws Exception if the input format was not set
     */
    public boolean input(Instance instance) {
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }
        if (getOutputFormat() == null) {
            // create new Instances
            int len = m_AA.length() * m_AA.length();
            FastVector attInfo = new FastVector(len + 1);
            for (int i = 0; i < len; i++) {
                attInfo.addElement(new Attribute("a" + i));
            }
            Attribute labelAttr = (Attribute) getInputFormat().classAttribute().copy();
            attInfo.addElement(labelAttr);
            Instances instances = new Instances(instance.dataset().relationName() + "-" + toString(), attInfo, 0);
            instances.setClassIndex(instances.numAttributes() - 1);
            setOutputFormat(instances);
        }
        if (m_NewBatch) {
            resetQueue();
            m_NewBatch = false;
        }
        convertInstance(instance);
        return true;
    }

    /**
     * Signify that this batch of input to the filter is finished. If the filter
     * requires all instances prior to filtering, output() may now be called to
     * retrieve the filtered instances.
     *
     * @return true if there are instances pending output
     * @throws IllegalStateException
     *             if no input structure has been defined
     */
    public boolean batchFinished() {
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }

        // Convert pending input instances
        for (int i = 0; i < getInputFormat().numInstances(); i++) {
            convertInstance(getInputFormat().instance(i));
        }

        // Free memory
        flushInput();

        m_NewBatch = true;
        return (numPendingOutput() != 0);
    }

    /**
     * Converts an input instance into output.
     * @param instance  The input instance.
     */
    private void convertInstance(Instance instance) {
        Instance tmp = new Instance(getOutputFormat().numAttributes());
        tmp.setDataset(getOutputFormat());
        String peptide = instance.stringValue(0);
        String s = dipeptides(peptide);
        StringTokenizer st = new StringTokenizer(s, " ", false);
        int x = 0;
        while (st.hasMoreTokens()) {
            tmp.setValue(x++, Double.parseDouble(st.nextToken()));
        }
        tmp.setClassValue(instance.classValue());
        push(tmp);
    }

    /**
     * Computes dipeptides compositions
     *
     * @param seq  input sequence
     * @return   dipeptide compositions of the input sequence as a space-separated string
     */
    private String dipeptides(String seq) {
        String val = new String();
        double len = seq.length();

        NumberFormat format = new DecimalFormat("##.##");
        for (int i = 0; i < m_AA.length(); i++) {
            for (int j = 0; j < m_AA.length(); j++) {
                String s = m_AA.substring(i, i + 1) + m_AA.substring(j, j + 1);
                int x = -1;
                double count = 0;
                while ((x = seq.indexOf(s, x + 1)) >= 0) {
                    count++;
                }
                val += " " + format.format((double) (count / (len - 1)));
            }
        }

        return val.trim();
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
     *
     * @return A string description of this filter and its parameters.
     */
    public String toString() {
        return "SequenceComposition -A " + m_AA;
    }

    /**
     * Main method for testing this class.
     *
     * @param argv
     *            should contain arguments to the filter: use -h for help
     */
    public static void main(String[] argv) {
        runFilter(new SequenceDiCompositions(), argv);
    }
}
