

import java.awt.*;
import javax.swing.*;
import java.beans.*;
import java.io.*;
import java.text.DecimalFormat;
import java.util.Vector;
import weka.core.*;
import weka.classifiers.*;
import javax.swing.filechooser.FileFilter;
import iubio.bioseq.*;
import iubio.readseq.*;


/**
 * Predictor application
 */
public class PredictorJInternalFrame extends javax.swing.JInternalFrame
        implements PropertyChangeListener {





    protected JFileChooser m_ModelFile;     // Model file
    protected JFileChooser m_TestFile;      // Test file
    protected JFileChooser m_OutputFile;    // Output file
    protected Classifier m_Classifier = null;   // could be FilteredClassifier
    protected Instances m_Header = null;
    protected boolean m_IsClassification = true;
    protected int m_PepLength = 20;         // default peptide length
    // reader and writer
    protected BufferedReader m_Reader = null;
    protected Writer m_Writer = null;
    protected DecimalFormat m_Formatter = new DecimalFormat();
    protected String m_OldPepLen = "2";
    private Task task;

    //-------------- Inner class Task -----------------------------------
    class Task extends javax.swing.SwingWorker<Void, Void> {
        /*
         * Main task. Executed in back    public PredictorJInternalFrame(YoussefbakrApp aThis) {
        throw new UnsupportedOperationException("Not yet implemented");
        }

        ground thread.
         */
        // @Override

        public Void doInBackground() {
            int K = m_PepLength / 2;
            String prfx = new String();
            for (int i = 0; i < K; i++) {
                prfx += "X";
            }

            StringBuffer predictions = new StringBuffer();
            String header;
            if (cmbInstType.getSelectedIndex() == 0) {
                header = "ID\t Position \t Peptide\t Score\n";
            } else {
                header = "ID\t Position \t Residue\t Score\n";
            }

            if (cmbInFormat.getSelectedIndex() == 1) { // test set is a list of peptides
                String line = null;
                Vector epitopes = new Vector();
                try {
                    while ((line = m_Reader.readLine()) != null) {
                        epitopes.add(line.trim());
                    }// end while

                    int size = epitopes.size();
                    prgBar.setMaximum(0);
                    prgBar.setMaximum(size);
                    prgBar.setValue(0);
                    prgBar.setVisible(true);
                    for (int i = 1; !isCancelled() && i <= size; i++) {
                        line = (String) epitopes.elementAt(i - 1);
                        if (cmbInstType.getSelectedIndex() == 0) {   // peptide based
                            if (m_PepLength == -1) {
                                predictions.append(predictPeptides("" + i, line, line.length()));
                            } else {
                                predictions.append(predictPeptides("" + i, line, m_PepLength));
                            }
                        } else {   // residue based
                            line = prfx + line.trim() + prfx;
                            predictions.append(predictWindows("" + i, line.trim(), m_PepLength));
                        }
                        prgBar.setValue(i);
                    }//end i loop
                } catch (Exception e) {
                    JOptionPane.showMessageDialog(null, "Test data does not match training data. \n Please check Peptide/Window length.\n" + e.toString());
                    return null;
                }


            } else {   // process fasta sequences
                try {
                    // your data goes here
                    Object inputObject = new FileReader(m_TestFile.getSelectedFile());
                    Readseq rd = new Readseq();
                    rd.setInputObject(inputObject);
                    Vector antigens = new Vector();
                    Vector IDs = new Vector();

                    if (rd.isKnownFormat() && rd.readInit()) {
                        while (rd.readNext()) {
                            BioseqRecord seqrec = new BioseqRecord(rd.nextSeq());
                            Bioseq sequence = seqrec.getseq();
                            antigens.add(seqrec.getID());
                            IDs.add(sequence.toString());
                        } //end while
                        rd.close();
                        int size = antigens.size();
                        prgBar.setMinimum(0);
                        prgBar.setMaximum(size);
                        prgBar.setValue(0);
                        prgBar.setVisible(true);
                        for (int i = 1; !isCancelled() && i <= size; i++) {
                            String seqName = (String) antigens.elementAt(i - 1);
                            String strSeq = (String) IDs.elementAt(i - 1);
                            if (cmbInstType.getSelectedIndex() == 1) {  // residue based
                                strSeq = prfx + strSeq + prfx;
                                predictions.append(predictWindows(seqName, strSeq, m_PepLength));
                            } else {
                                predictions.append(predictPeptides(seqName, strSeq, m_PepLength));
                            }
                            prgBar.setValue(i);
                        }
                    } else {
                        rd.close();
                        JOptionPane.showMessageDialog(null, "Test data is not in a valid fasta format. \n");
                        return null;
                    }
                } catch (Exception e) {
                    JOptionPane.showMessageDialog(null, "Test data does not match training data. \n Please check Peptide/Window length.\n" + e.toString());
                    return null;
                }
            }// end else fasta sequences
            txtOutArea.append(header);
            // ouput predictions
            txtOutArea.append(predictions.toString());
            try {
                m_Writer.write(header);
                m_Writer.write(predictions.toString());
                m_Writer.close();
                m_Reader.close();
            } catch (Exception e) {
                JOptionPane.showMessageDialog(null, "Error writing predictions to output file \n" + e.toString());
                return null;
            }

            return null;
        }

        /**
         * Peptide-based predictions.
         * @param seqName  sequence ID
         * @param sequence amino acid sequence
         * @param length    peptide length
         * @return predictions
         * @throws java.lang.Exception
         */
        private StringBuffer predictPeptides(String seqName, String sequence, int length) throws Exception {
            // check that the three input files have been identified
            StringBuffer str = new StringBuffer();
            int len = sequence.length();
            if (len < length) {
                str.append(seqName + "\t Error: epitope length is shorter than the specified length \n");
                return str;
            }

            // Note, we assume the index of the positive class is 1
            for (int start = 0; !isCancelled() && start <= len - length; start++) {	// for each epitope
                String peptide = sequence.substring(start, start + length);
                Instance newInst = new Instance(2);
                newInst.setDataset(m_Header);
                newInst.setValue(0, peptide);
                if (!m_IsClassification) {
                    str.append(seqName + "\t " + (int) (start + 1) + ":" + (int) (start + length) + "\t " + peptide + "\t " + m_Formatter.format(m_Classifier.classifyInstance(newInst)) + "\n");
                } else {
                    str.append(seqName + "\t " + (int) (start + 1) + ":" + (int) (start + length) + "\t " + peptide + "\t " + m_Formatter.format(m_Classifier.distributionForInstance(newInst)[1]) + "\n");
                }
            }

            return str;
        }

        /**
         * Residue-based predictions.
         * @param seqName sequence ID
         * @param sequence amino acid sequence
         * @param length    window size
         * @return predictions
         * @throws java.lang.Exception
         */
        private StringBuffer predictWindows(String seqName, String sequence, int length) throws Exception {
            // check that the three input files have been identified
            StringBuffer str = new StringBuffer();
            int len = sequence.length();
            if (len < length) {
                str.append(seqName + "\t Error: window length is shorter than the specified length \n");
                return str;
            }

            // Note, we assume the index of the positive class is 1
            int K = length / 2;

            for (int start = 0; !isCancelled() && start <= len - length; start++) {	// for each epitope
                String peptide = sequence.substring(start, start + length);
                Instance newInst = new Instance(2);
                newInst.setDataset(m_Header);
                newInst.setValue(0, peptide);
                if (!m_IsClassification) {
                    str.append(seqName + "\t " + (int) (start + 1) + "\t " + peptide.substring(K, K + 1) + "\t " + m_Formatter.format(m_Classifier.classifyInstance(newInst)) + "\n");
                } else {
                    str.append(seqName + "\t " + (int) (start + 1) + "\t " + peptide.substring(K, K + 1) + "\t " + m_Formatter.format(m_Classifier.distributionForInstance(newInst)[1]) + "\n");
                }
            }
            return str;
        }


        /*
         * Executed in event dispatching thread
         */
        //@Override
        public void done() {
            btnPredict.setEnabled(true);
            btnStop.setEnabled(false);
            prgBar.setVisible(false);
            lblStatus.setText("  Done");
            setCursor(null); //turn off the wait cursor
        }
    }
    //-------------- End inner class Task -------------------------------

    // ------------- Inner class ExtensionFileFilter --------------------
    class ExtensionFileFilter extends FileFilter {

        String description;
        String extensions[];

        public ExtensionFileFilter(String description, String extension) {
            this(description, new String[]{extension});
        }

        public ExtensionFileFilter(String description, String extensions[]) {
            if (description == null) {
                this.description = extensions[0];
            } else {
                this.description = description;
            }
            this.extensions = (String[]) extensions.clone();
            toLower(this.extensions);
        }

        private void toLower(String array[]) {
            for (int i = 0, n = array.length; i < n; i++) {
                array[i] = array[i].toLowerCase();
            }
        }

        public String getDescription() {
            return description;
        }

        public boolean accept(File file) {
            if (file.isDirectory()) {
                return true;
            } else {
                String path = file.getAbsolutePath().toLowerCase();
                for (int i = 0, n = extensions.length; i < n; i++) {
                    String extension = extensions[i];
                    if ((path.endsWith(extension) && (path.charAt(path.length() - extension.length() - 1)) == '.')) {
                        return true;
                    }
                }
            }
            return false;
        }
    }
    // ------------ End Inner class ExtensionFileFilter -----------------

    /** Creates new form PredictorJInternalFrame */
    public PredictorJInternalFrame() {

        initComponents();
        m_ModelFile = new javax.swing.JFileChooser();
        m_ModelFile.setCurrentDirectory(new File("."));
        FileFilter filter = new ExtensionFileFilter("Model", new String[]{"model"});
        m_ModelFile.setFileFilter(filter);

        m_TestFile = new javax.swing.JFileChooser();
        m_TestFile.setCurrentDirectory(new File("."));

        m_OutputFile = new javax.swing.JFileChooser();
        m_OutputFile.setCurrentDirectory(new File("."));

        m_Formatter.setMinimumFractionDigits(3);
        prgBar.setStringPainted(true);
    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        btnGFormat = new javax.swing.ButtonGroup();
        pnlPara = new javax.swing.JPanel();
        btnModel = new javax.swing.JButton();
        btnTestData = new javax.swing.JButton();
        btnOutFile = new javax.swing.JButton();
        txtModel = new javax.swing.JTextField();
        txtTest = new javax.swing.JTextField();
        txtOutFile = new javax.swing.JTextField();
        btnPredict = new javax.swing.JButton();
        btnStop = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        txtOutArea = new javax.swing.JTextArea();
        jPanel1 = new javax.swing.JPanel();
        cmbInFormat = new javax.swing.JComboBox();
        txtEpLen = new javax.swing.JTextField();
        cmbInstType = new javax.swing.JComboBox();
        jLabel1 = new javax.swing.JLabel();
        jLabel2 = new javax.swing.JLabel();
        jLabel3 = new javax.swing.JLabel();
        jPanel3 = new javax.swing.JPanel();
        lblStatus = new javax.swing.JLabel();
        prgBar = new javax.swing.JProgressBar();

        setClosable(true);
        setIconifiable(true);
        setMaximizable(true);
        setResizable(true);
        setTitle("Epitopes Prediction Software"); // NOI18N

        btnModel.setText("Model");
        btnModel.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnModelActionPerformed(evt);
            }
        });

        btnTestData.setText("Test data");
        btnTestData.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnTestDataActionPerformed(evt);
            }
        });

        btnOutFile.setText("Output file");
        btnOutFile.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnOutFileActionPerformed(evt);
            }
        });

        txtModel.setEditable(false);
        txtModel.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                txtModelActionPerformed(evt);
            }
        });

        txtTest.setEditable(false);

        txtOutFile.setEditable(false);

        btnPredict.setFont(new java.awt.Font("Tahoma", 0, 12));
        btnPredict.setText("Predict");
        btnPredict.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnPredictActionPerformed(evt);
            }
        });

        btnStop.setFont(new java.awt.Font("Tahoma", 0, 12));
        btnStop.setText("Stop");
        btnStop.setEnabled(false);
        btnStop.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnStopActionPerformed(evt);
            }
        });

        org.jdesktop.layout.GroupLayout pnlParaLayout = new org.jdesktop.layout.GroupLayout(pnlPara);
        pnlPara.setLayout(pnlParaLayout);
        pnlParaLayout.setHorizontalGroup(
            pnlParaLayout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(org.jdesktop.layout.GroupLayout.TRAILING, pnlParaLayout.createSequentialGroup()
                .addContainerGap()
                .add(pnlParaLayout.createParallelGroup(org.jdesktop.layout.GroupLayout.TRAILING)
                    .add(org.jdesktop.layout.GroupLayout.LEADING, btnPredict, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 195, Short.MAX_VALUE)
                    .add(org.jdesktop.layout.GroupLayout.LEADING, btnOutFile, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 195, Short.MAX_VALUE)
                    .add(org.jdesktop.layout.GroupLayout.LEADING, btnModel, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 195, Short.MAX_VALUE)
                    .add(btnTestData, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 195, Short.MAX_VALUE))
                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                .add(pnlParaLayout.createParallelGroup(org.jdesktop.layout.GroupLayout.TRAILING)
                    .add(pnlParaLayout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
                        .add(txtModel, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 212, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                        .add(txtTest, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 212, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                        .add(txtOutFile, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 212, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE))
                    .add(btnStop, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 212, Short.MAX_VALUE))
                .addContainerGap())
        );
        pnlParaLayout.setVerticalGroup(
            pnlParaLayout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(pnlParaLayout.createSequentialGroup()
                .addContainerGap()
                .add(pnlParaLayout.createParallelGroup(org.jdesktop.layout.GroupLayout.CENTER)
                    .add(txtModel, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 29, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                    .add(btnModel, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 34, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                .add(pnlParaLayout.createParallelGroup(org.jdesktop.layout.GroupLayout.CENTER)
                    .add(btnTestData, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 34, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                    .add(txtTest, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 29, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                .add(pnlParaLayout.createParallelGroup(org.jdesktop.layout.GroupLayout.CENTER)
                    .add(txtOutFile, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 29, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                    .add(btnOutFile, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 34, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED, 46, Short.MAX_VALUE)
                .add(pnlParaLayout.createParallelGroup(org.jdesktop.layout.GroupLayout.BASELINE)
                    .add(btnPredict, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 31, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                    .add(btnStop, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 31, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)))
        );

        jScrollPane1.setBorder(javax.swing.BorderFactory.createTitledBorder("Predictions"));

        txtOutArea.setColumns(20);
        txtOutArea.setEditable(false);
        txtOutArea.setFont(new java.awt.Font("Courier New", 0, 14));
        txtOutArea.setRows(5);
        jScrollPane1.setViewportView(txtOutArea);

        cmbInFormat.setModel(new javax.swing.DefaultComboBoxModel(new String[] { "  fasta sequences", "  epitopes list" }));
        cmbInFormat.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cmbInFormatActionPerformed(evt);
            }
        });

        txtEpLen.setText("20");

        cmbInstType.setModel(new javax.swing.DefaultComboBoxModel(new String[] { "  peptide based", "  residue based" }));

        jLabel1.setText("Input format");

        jLabel2.setText("Instance type");

        jLabel3.setText("Peptide/Window length");

        org.jdesktop.layout.GroupLayout jPanel1Layout = new org.jdesktop.layout.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(jPanel1Layout.createSequentialGroup()
                .addContainerGap()
                .add(jPanel1Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING, false)
                    .add(jLabel3, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .add(jLabel2, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .add(jLabel1, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .add(19, 19, 19)
                .add(jPanel1Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
                    .add(jPanel1Layout.createSequentialGroup()
                        .add(txtEpLen, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 38, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                        .addContainerGap())
                    .add(org.jdesktop.layout.GroupLayout.TRAILING, jPanel1Layout.createSequentialGroup()
                        .add(jPanel1Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.TRAILING, false)
                            .add(org.jdesktop.layout.GroupLayout.LEADING, cmbInstType, 0, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .add(org.jdesktop.layout.GroupLayout.LEADING, cmbInFormat, 0, 145, Short.MAX_VALUE))
                        .add(80, 80, 80))))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(jPanel1Layout.createSequentialGroup()
                .add(jPanel1Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
                    .add(jPanel1Layout.createSequentialGroup()
                        .add(7, 7, 7)
                        .add(cmbInFormat, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 29, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE))
                    .add(org.jdesktop.layout.GroupLayout.TRAILING, jLabel1, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 36, Short.MAX_VALUE))
                .addPreferredGap(org.jdesktop.layout.LayoutStyle.UNRELATED)
                .add(jPanel1Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.BASELINE)
                    .add(cmbInstType, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 29, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                    .add(jLabel2, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 28, Short.MAX_VALUE))
                .addPreferredGap(org.jdesktop.layout.LayoutStyle.UNRELATED)
                .add(jPanel1Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.BASELINE)
                    .add(txtEpLen, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 29, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                    .add(jLabel3, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 28, Short.MAX_VALUE))
                .addContainerGap())
        );

        jPanel3.setBorder(javax.swing.BorderFactory.createTitledBorder("Status"));

        lblStatus.setText(" Welcome");

        org.jdesktop.layout.GroupLayout jPanel3Layout = new org.jdesktop.layout.GroupLayout(jPanel3);
        jPanel3.setLayout(jPanel3Layout);
        jPanel3Layout.setHorizontalGroup(
            jPanel3Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(org.jdesktop.layout.GroupLayout.TRAILING, jPanel3Layout.createSequentialGroup()
                .addContainerGap()
                .add(lblStatus, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 294, Short.MAX_VALUE)
                .add(18, 18, 18)
                .add(prgBar, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 485, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE))
        );
        jPanel3Layout.setVerticalGroup(
            jPanel3Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(jPanel3Layout.createSequentialGroup()
                .add(jPanel3Layout.createParallelGroup(org.jdesktop.layout.GroupLayout.CENTER)
                    .add(lblStatus, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 26, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                    .add(prgBar, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        org.jdesktop.layout.GroupLayout layout = new org.jdesktop.layout.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(layout.createSequentialGroup()
                .add(pnlPara, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                .add(18, 18, 18)
                .add(jPanel1, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 372, Short.MAX_VALUE)
                .addContainerGap())
            .add(layout.createSequentialGroup()
                .addContainerGap()
                .add(jPanel3, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
            .add(layout.createSequentialGroup()
                .addContainerGap()
                .add(jScrollPane1, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 823, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
            .add(layout.createSequentialGroup()
                .add(layout.createParallelGroup(org.jdesktop.layout.GroupLayout.LEADING)
                    .add(pnlPara, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)
                    .add(layout.createSequentialGroup()
                        .add(11, 11, 11)
                        .add(jPanel1, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE)))
                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                .add(jScrollPane1, org.jdesktop.layout.GroupLayout.DEFAULT_SIZE, 357, Short.MAX_VALUE)
                .addPreferredGap(org.jdesktop.layout.LayoutStyle.RELATED)
                .add(jPanel3, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE, 56, org.jdesktop.layout.GroupLayout.PREFERRED_SIZE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void btnModelActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnModelActionPerformed
        // TODO add your handling code here:
        int option = m_ModelFile.showOpenDialog(this);
        if (option == JFileChooser.APPROVE_OPTION) {
            txtModel.setText(m_ModelFile.getSelectedFile().getName());
        }
    }//GEN-LAST:event_btnModelActionPerformed

    private void btnTestDataActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnTestDataActionPerformed
        // TODO add your handling code here:
        int option = m_TestFile.showOpenDialog(this);
        if (option == JFileChooser.APPROVE_OPTION) {
            txtTest.setText(m_TestFile.getSelectedFile().getName());
        }
    }//GEN-LAST:event_btnTestDataActionPerformed

    private void btnOutFileActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnOutFileActionPerformed
        // TODO add your handling code here:
        //int option = m_OutputFile.showOpenDialog(this);
        int option = m_OutputFile.showSaveDialog(this);
        if (option == JFileChooser.APPROVE_OPTION) {
            txtOutFile.setText(m_OutputFile.getSelectedFile().getName());
        }
    }//GEN-LAST:event_btnOutFileActionPerformed

    private void txtModelActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_txtModelActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_txtModelActionPerformed

    private void btnPredictActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnPredictActionPerformed
        // TODO add your handling code here:
        txtOutArea.setText("");     // clear output

        if (txtModel.getText().equals("")) {
            JOptionPane.showMessageDialog(null, "Please specify the model file");
            btnModel.requestFocus();
        } else if (txtTest.getText().equals("")) {
            JOptionPane.showMessageDialog(null, "Please specify the test data file");
            btnTestData.requestFocus();
        } else if (txtOutFile.getText().equals("")) {
            JOptionPane.showMessageDialog(null, "Please specify the output file");
            btnOutFile.requestFocus();
        } else {
            // open test data file
            try {
                InputStreamReader is = new InputStreamReader(new FileInputStream(m_TestFile.getSelectedFile()));
                m_Reader = new BufferedReader(is);
            } catch (Exception e) {
                JOptionPane.showMessageDialog(null, "Error Opening test data file \n" + e.toString());
                return;
            }

            // open output file
            try {
                m_Writer = new FileWriter(m_OutputFile.getSelectedFile());
            } catch (Exception e) {
                JOptionPane.showMessageDialog(null, "Error Opening output file \n" + e.toString());
                return;
            }

            // load the model
            try {
                loadModel();
            } catch (Exception e) {
                JOptionPane.showMessageDialog(null, "Problem loading the model \n" + e.toString());
                lblStatus.setText("  Model Loading Failed");
                return;
            }

            m_PepLength = Integer.parseInt(txtEpLen.getText());
            if (m_PepLength == 0) {
                JOptionPane.showMessageDialog(null, "Illegal epitope length \n");
                return;
            }
            if (m_PepLength % 2 == 0 & cmbInstType.getSelectedIndex() == 1) {
                JOptionPane.showMessageDialog(null, "Window size should be an odd number \n");
                return;
            }
            btnPredict.setEnabled(false);
            lblStatus.setText("  Predicting");
            setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
            task = new Task();
            task.addPropertyChangeListener(this);
            task.execute();
            btnStop.setEnabled(true);
        }// end else
    }//GEN-LAST:event_btnPredictActionPerformed

    public void propertyChange(PropertyChangeEvent evt) {
        if ("progress" == evt.getPropertyName()) {
            int progress = (Integer) evt.getNewValue();
            prgBar.setValue(progress);
        }
    }
    private void cmbInFormatActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cmbInFormatActionPerformed
        // TODO add your handling code here:
        if (cmbInFormat.getSelectedIndex() == 1) { //current selection is list of peptides
            m_OldPepLen = txtEpLen.getText();
            txtEpLen.setText("-1");
            //txtEpLen.setEditable(false);
            cmbInstType.setSelectedIndex(0);
            //cmbInstType.setEnabled(false);
        } else {
            if (txtEpLen.getText().equals("-1")) {
                //txtEpLen.setEditable(true);
                txtEpLen.setText(m_OldPepLen);
                cmbInstType.setEnabled(true);
            }
        }
    }//GEN-LAST:event_cmbInFormatActionPerformed

    private void btnStopActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnStopActionPerformed
        // TODO add your handling code here:
        if (task.cancel(true)) {
            lblStatus.setText("  Canceled");
            btnStop.setEnabled(false);
        }
}//GEN-LAST:event_btnStopActionPerformed

    /**
     * loads the model
     * @throws java.lang.Exception
     */
    private void loadModel() throws Exception {
        lblStatus.setText("  Loading Model");
        ObjectInputStream model = new ObjectInputStream(new FileInputStream(m_ModelFile.getSelectedFile()));
        m_Classifier = (Classifier) model.readObject();
        // write model summary

        try {
            m_Header = (Instances) model.readObject();
        } catch (Exception e) {
            JOptionPane.showMessageDialog(null, e, "Load Failed", JOptionPane.ERROR_MESSAGE);
            m_Classifier = null;
            return;
        }
        StringBuffer outStr = new StringBuffer();
        outStr.append("------- Model Summary --------------\n");
        outStr.append("Loading model from " + m_ModelFile.getSelectedFile().getName() + "\n");
        outStr.append("Model scheme:  " + m_Classifier.getClass().getName() + " ");
        if (m_Classifier instanceof OptionHandler) {
            String[] ops = ((OptionHandler) m_Classifier).getOptions();
            outStr.append(" " + Utils.joinOptions(ops) + "\n");
        }
        if (m_Header != null) {
            outStr.append("Relation : " + m_Header.relationName() + "\n");
            if (m_Header.classAttribute().isNumeric()) {
                m_IsClassification = false;
            }
        }
        outStr.append("------- End Model Summary -----------\n\n");
        txtOutArea.setText("");  // empty
        txtOutArea.append(outStr.toString());
        m_Writer.write(outStr.toString());
        model.close();
    }
    


    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.ButtonGroup btnGFormat;
    private javax.swing.JButton btnModel;
    private javax.swing.JButton btnOutFile;
    private javax.swing.JButton btnPredict;
    private javax.swing.JButton btnStop;
    private javax.swing.JButton btnTestData;
    private javax.swing.JComboBox cmbInFormat;
    private javax.swing.JComboBox cmbInstType;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel3;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JLabel lblStatus;
    private javax.swing.JPanel pnlPara;
    private javax.swing.JProgressBar prgBar;
    private javax.swing.JTextField txtEpLen;
    private javax.swing.JTextField txtModel;
    private javax.swing.JTextArea txtOutArea;
    private javax.swing.JTextField txtOutFile;
    private javax.swing.JTextField txtTest;
    // End of variables declaration//GEN-END:variables
}
