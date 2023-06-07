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
 *    SequenceToNominal.java
 *    Copyright (C) 2009 Yasser EL-Manzalawy
 *
 */

package epit.filters.unsupervised.attribute;

import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.UnsupervisedFilter;
import java.util.*;

/**
 * <!-- globalinfo-start -->
 * Converts a fixed-length sequence attribute defined over an alphabet A into |A| nominal attributes
 * where each attribute can take any symbol in A. The output of this filter can be passed
 * to a NominalToBinary filter to get the 0/1 representation of a protein sequence.
 * <p/> <!-- globalinfo-end  -->
 *
 * <!-- options-start --> Valid options are: <p/>
 *
 * <pre>
 * -A &lt;string&gt;
 *  The alphabet (default standard 20 amino acids).
 * </pre>
 *
 * <pre>
 * -L &lt;num&gt;
 *  The sequence length (default 9).
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Len Trigg (len@reeltwo.com)
 * @version $Revision: 1.6 $
 */
public class SequenceToNominal extends Filter implements UnsupervisedFilter,
        OptionHandler {

    /** for serialization */
    static final long serialVersionUID = 3119607037607101160L;
    /** The sequence alphabet (default = standard 20 amino acids). */
    protected String m_AA = "ACDEFGHIKLMNPQRSTVWYX"; // X is included
    /** Sequence length */
    protected int m_Length = 9;
    private int strIndex = 0;

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
     * Sets the Length parameter tip text in the Weka GUI.
     * @return tip text.
     */
    public String lengthTipText() {
        return "Length of each peptide";
    }

    /**
     * Gets the sequence length.
     * @return The sequence length.
     */
    public int getLength() {
        return m_Length;
    }

    /**
     * Sets the sequence length.
     * @param alphabet The sequence length.
     */
    public void setLength(int len) {
        m_Length = len;
    }

    /**
     * Lists valid options for that filter.
     * <!-- options-start --> Valid options are: <p/>
     *
     * <pre>
     * -A &lt;string&gt;
     *  The alphabet (default standard 20 amino acids).
     * </pre>
     *
     * <pre>
     * -L &lt;num&gt;
     *  The sequence length (default 9).
     * </pre>
     *
     * <!-- options-end -->
     *
     */
    public Enumeration listOptions() {

        Vector newVector = new Vector(2);

        newVector.addElement(new Option("\tSets the peptide length\n", "L", 1,
                "-L <int>"));
        newVector.addElement(new Option("\tSets sequence alphabet\n", "A", 1,
                "-A <string>"));

        return newVector.elements();
    }

    /**
     * Sets filter options.
     * -A sequence alphabet (default 20 amino acids).
     * -L sequence length  (default 9).
     */
    public void setOptions(String[] options) throws Exception {
        String tmpStr = Utils.getOption('L', options);
        if (tmpStr.length() != 0) {
            setLength(Integer.parseInt(tmpStr));
        } else {
            setLength(9);
        }

        tmpStr = Utils.getOption('A', options);
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

        String[] options = new String[4];
        int current = 0;
        options[current++] = "-L";
        options[current++] = "" + getLength();
        options[current++] = "-A";
        options[current++] = "" + getAlphabet();
        return options;
    }

    /**
     * Returns a string describing this classifier
     *
     * @return a description of the classifier suitable for displaying in the
     *         explorer/experimenter gui
     */
    public String globalInfo() {
        return "SequenceToNominal Filter by Yasser EL-Manzalawy.";
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
        // setOutputFormat(instanceInfo);
        m_FirstBatchDone = false;

        // get the index of the string attribute
        Enumeration e = instanceInfo.enumerateAttributes();
        strIndex = 0;
        while (e.hasMoreElements()) {
            Attribute at = (Attribute) e.nextElement();
            if (at.isString()) {
                break;
            }
            strIndex++;
        }

        FastVector attInfo = new FastVector(m_Length + instanceInfo.numAttributes() - 1);
        FastVector nominalAtt = new FastVector(m_AA.length() + 1);
        for (int a = 0; a < m_AA.length(); a++) {
            nominalAtt.addElement(m_AA.substring(a, a + 1));
        }

        for (int i = 0; i < strIndex; i++) {
            attInfo.addElement(instanceInfo.attribute(i).copy());
        }

        for (int i = 0; i < m_Length; i++) {
            attInfo.addElement(new Attribute("a" + i, nominalAtt));
        }
        for (int i = strIndex + 1; i < instanceInfo.numAttributes(); i++) {
            attInfo.addElement(instanceInfo.attribute(i).copy());
        }
        Instances outputFormat = new Instances("nominalData", attInfo, 0);
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
            Enumeration e = instance.enumerateAttributes();
            strIndex = 0;
            while (e.hasMoreElements()) {
                Attribute at = (Attribute) e.nextElement();
                if (at.isString()) {
                    break;
                }
                strIndex++;
            }

            FastVector attInfo = new FastVector(m_Length + instance.numAttributes() - 1);
            FastVector nominalAtt = new FastVector(m_AA.length() + 1);
            for (int a = 0; a < m_AA.length(); a++) {
                nominalAtt.addElement(m_AA.substring(a, a + 1));
            }

            for (int i = 0; i < strIndex; i++) {
                attInfo.addElement(instance.attribute(i).copy());
            }

            for (int i = 0; i < m_Length; i++) {
                attInfo.addElement(new Attribute("a" + i, nominalAtt));
            }
            for (int i = strIndex + 1; i < instance.numAttributes(); i++) {
                attInfo.addElement(instance.attribute(i).copy());
            }
            Instances outputFormat = new Instances("nominalData", attInfo, 0);
            outputFormat.setClassIndex(outputFormat.numAttributes() - 1);
            setOutputFormat(outputFormat);

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

    private void convertInstance(Instance instance) {
        Instance tmp = new Instance(getOutputFormat().numAttributes());
        tmp.setDataset(getOutputFormat());

        int count = 0;
        for (int i = 0; i < strIndex; i++) {
            tmp.setValue(i, instance.value(i));
            count++;
        }

        String peptide = instance.stringValue(strIndex);

        for (int i = 0; i < m_Length; i++) {
            tmp.setValue(count++, peptide.substring(i, i + 1)); // may throw Exception if not a standard aa or X
        }

        for (int i = strIndex + 1; i < instance.numAttributes(); i++) {
            tmp.setValue(count++, instance.value(i));
        }
        push(tmp);
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
     *            should contain arguments to the filter: use -h for help
     */
    public static void main(String[] argv) {
        runFilter(new SequenceToNominal(), argv);
    }
}
