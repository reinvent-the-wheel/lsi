package com.github.reinvent.the.wheel.lsi;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.*;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static java.lang.Math.min;

public class Lsi {
    private WordScoreCalc wordScoreCalc = (String word, int wordCountInDoc, int allWordCountInDoc) -> (wordCountInDoc + 0.0) / allWordCountInDoc;
    private Dict dict;
    int num_topics = 300;

    //
    //add_documents -> stochastic_svd
    public static void stochastic_svd() {

    }
    public RealVector getVector(String[] document) {
        List<String[]> documents = new ArrayList<>();
        documents.add(document);
        return getVector(documents).getRowVector(0);
    }
    public RealMatrix getVector(List<String[]> documents) {
        int num_terms = dict.size();
        OpenMapRealMatrix wordDocumentScoreMatrix = new OpenMapRealMatrix(num_terms, documents.size());
        for (int i = 0; i < documents.size(); i++) {
            String[] document = documents.get(i);
            Map<String, AtomicInteger> wordCountMap = new HashMap<>();
            for (String w : document) {
                wordCountMap.computeIfAbsent(w, k -> new AtomicInteger(0)).incrementAndGet();
            }
            int finalI = i;
            wordCountMap.forEach((word, count) -> {
                Integer index = dict.getIndex(word);
                if(index!=null){
                    double score = wordScoreCalc.score(word, count.get(), wordCountMap.size());
                    wordDocumentScoreMatrix.setEntry(index, finalI, score);
                }
            });
        }
//        topic_dist = (vec.T * self.projection.u[:, :self.num_topics]).T  # (x^T * u).T = u^-1 * x
        RealMatrix topic_dist = wordDocumentScoreMatrix.transpose().multiply(this.u.getSubMatrix(0, this.u.getRowDimension() - 1, 0, min(this.u.getColumnDimension() - 1, this.num_topics))).transpose();
        return topic_dist.transpose();
    }

    public void addDocuments(List<String[]> documents) {
        if (documents.size() == 0) return;
        int num_terms = dict.size();
        int rank = num_topics;
        int extra_dims = 100;
        double decay = 1.0;
        double eps = 1e-6;

        OpenMapRealMatrix wordDocumentScoreMatrix = new OpenMapRealMatrix(num_terms, documents.size());
        for (int i = 0; i < documents.size(); i++) {
            String[] document = documents.get(i);
            Map<String, AtomicInteger> wordCountMap = new HashMap<>();
            for (String w : document) {
                wordCountMap.computeIfAbsent(w, k -> new AtomicInteger(0)).incrementAndGet();
            }
            int finalI = i;
            wordCountMap.forEach((word, count) -> {
                Integer index = dict.getIndex(word);
                if(index!=null){
                    double score = wordScoreCalc.score(word, count.get(), wordCountMap.size());
                    wordDocumentScoreMatrix.setEntry(index, finalI, score);
                }
            });
        }
        int samples = rank + extra_dims;
//        RealMatrix y =  MatrixUtils.createRealMatrix(num_terms,samples);
        RealMatrix o = normal(0.0, 1.0, documents.size(), samples);
        RealMatrix y = wordDocumentScoreMatrix.multiply(o);
        RealMatrix q = qr_destroy(y).getQ();
        int power_iters = 2;
        for (int power_iter = 0; power_iter < power_iters; power_iter++) {
            q = wordDocumentScoreMatrix.transpose().multiply(q);
            q = wordDocumentScoreMatrix.multiply(q);
            q = qr_destroy(q).getQ();
        }

        RealMatrix qt = q.getSubMatrix(0, q.getRowDimension() - 1, 0, min(samples - 1, q.getColumnDimension() - 1));
        q = null;
        RealMatrix b = qt.multiply(wordDocumentScoreMatrix);
        SingularValueDecomposition singularValueDecomposition = new SingularValueDecomposition(b);
        RealMatrix u = singularValueDecomposition.getU();
        RealVector s = MatrixUtils.createRealVector(singularValueDecomposition.getSingularValues());
        singularValueDecomposition = null;
        q = qt.transpose();
        qt = null;

        int keep = clip_spectrum(s.ebeMultiply(s), rank, eps);
        u = u.getSubMatrix(0, u.getRowDimension() - 1, 0, min(u.getColumnDimension() - 1, keep - 1));
        s = s.getSubVector(0, min(keep, s.getDimension()));
        u = q.multiply(u);
//
        u = u.getSubMatrix(0, u.getRowDimension() - 1, 0, min(u.getColumnDimension() - 1, num_topics - 1));
        s = s.getSubVector(0, min(num_topics, s.getDimension()));
//
        if (this.u == null || this.s == null) {
            this.u = u.copy();
            this.s = s.copy();
        } else {//merge
            int m = this.u.getRowDimension();
            int n1 = this.u.getColumnDimension();
            int n2 = u.getColumnDimension();
//            c = np.dot(self.u.T, other.u)
            RealMatrix c = this.u.transpose().multiply(u);
//            other.u -= np.dot(self.u, c)
            u = u.subtract(this.u.multiply(c));
            Qr qr = qr_destroy(u);
            q = qr.getQ();
            RealMatrix r = qr.getR();
//            np.diag(decay * self.s)
            RealMatrix p1 = diag(this.s.mapMultiply(decay));
//            np.multiply(c, other.s)
            RealMatrix p2 = c.multiply(diag(s));
//            matutils.pad(np.array([]).reshape(0, 0), min(m, n2), n1)
            RealMatrix p3 = MatrixUtils.createRealMatrix(min(m, n2), n1);
//            np.multiply(r, other.s)
            RealMatrix p4 = r.multiply(diag(s));
//            np.bmat([p1, p2], [p3, p4]])
            RealMatrix k1 = bmat(new RealMatrix[][]{{p1, p2}, {p3, p4}});
//            u_k, s_k, _ = scipy.linalg.svd(k, full_matrices=False)
            singularValueDecomposition = new SingularValueDecomposition(k1);
            RealMatrix u_k = singularValueDecomposition.getU();
            RealVector s_k = MatrixUtils.createRealVector(singularValueDecomposition.getSingularValues());
//            k = clip_spectrum(s_k**2, self.k)
            int k = clip_spectrum(s_k.ebeMultiply(s_k), num_topics, 0.001);
//            u1_k, u2_k, s_k = np.array(u_k[:n1, :k]), np.array(u_k[n1:, :k]), s_k[:k]
            RealMatrix u1_k = u_k.getSubMatrix(0, min(u_k.getRowDimension() - 1, n1 - 1), 0, min(u_k.getColumnDimension() - 1, k - 1));
            RealMatrix u2_k = u_k.getSubMatrix(n1, u_k.getRowDimension() - 1, 0, min(u_k.getColumnDimension() - 1, k - 1));
            s_k = s_k.getSubVector(0, min(s_k.getDimension(), k));
            this.s = s_k;
//            self.u = np.dot(self.u, u1_k)
            this.u = this.u.multiply(u1_k);
//            q = np.dot(q, u2_k)
            q = q.multiply(u2_k);
            this.u = this.u.add(q);
//            # make each column of U start with a non-negative number (to force canonical decomposition)
//            if self.u.shape[0] > 0:
//            for i in xrange(self.u.shape[1]):
//            if self.u[0, i] < 0.0:
//            self.u[:, i] *= -1.0
            if (this.u.getRowDimension() > 0) {
                for (int i = 0; i < u.getColumnDimension(); i++) {
                    if (this.u.getEntry(0, i) < 0.0) {
                        for (int j = 0; j < u.getRowDimension(); j++) {
                            this.u.setEntry(j, i, this.u.getEntry(j, i) * (-1.0));
                        }
                    }
                }
            }
        }
        print_topics(5);
    }

    private void print_topics(int i) {

    }

    private static RealMatrix bmat(RealMatrix[][] rows) {
        int r = Arrays.stream(rows).mapToInt(row -> row[0].getRowDimension()).sum();
        int c = Arrays.stream(rows[0]).mapToInt(column -> column.getColumnDimension()).sum();
        RealMatrix matrix = MatrixUtils.createRealMatrix(r, c);
        r = 0;
        for (RealMatrix[] row : rows) {
            for (int i = 0; i < row[0].getRowDimension(); i++) {
                c = 0;
                for (RealMatrix m : row) {
                    for (int j = 0; j < m.getColumnDimension(); j++) {
                        matrix.setEntry(r, c++, m.getEntry(i, j));
                    }
                }
                r++;
            }
        }
        return matrix;
    }

    private static RealMatrix diag(RealVector vector) {
        RealMatrix matrix = MatrixUtils.createRealMatrix(vector.getDimension(), vector.getDimension());
        for (int i = 0; i < vector.getDimension(); i++) {
            matrix.setEntry(i, i, vector.getEntry(i));
        }
        return matrix;
    }


    private Qr qr_destroy(RealMatrix y) {
        QRDecomposition qrDecomposition = new QRDecomposition(y);
        RealMatrix q = qrDecomposition.getQ();
        q = q.getSubMatrix(0, min(q.getRowDimension() - 1, y.getRowDimension() - 1),
                0, min(q.getColumnDimension() - 1, y.getColumnDimension() - 1)); //删除多余的列
        RealMatrix r = qrDecomposition.getR();
        return new Qr(q, r);
    }

    private static class Qr {
        private final RealMatrix q;
        private final RealMatrix r;

        public Qr(RealMatrix q, RealMatrix r) {
            this.q = q;
            this.r = r;
        }

        public RealMatrix getQ() {
            return q;
        }

        public RealMatrix getR() {
            return r;
        }
    }

    private RealMatrix u;
    private RealVector s;


    private static RealVector cumsum(RealVector vector) {
        AtomicReference<Double> sum = new AtomicReference<>();
        sum.set(0.0);
        return vector.copy().mapToSelf(x -> sum.getAndSet(sum.get() + x) + x);
    }

    private int clip_spectrum(RealVector s, int k, double discard) {
//        rel_spectrum = np.abs(1.0 - np.cumsum(s / np.sum(s)))
        RealVector rel_spectrum = cumsum(s.mapDivide(Arrays.stream(s.toArray()).sum())).mapMultiply(-1.0).mapAdd(1).copy().mapToSelf(x -> Math.abs(x));
//        small = 1 + len(np.where(rel_spectrum > min(discard, 1.0 / k))[0])
        double small = Arrays.stream(rel_spectrum.copy().mapToSelf(x -> x > min(discard, 1.0 / k) ? 1 : 0).toArray()).sum() + 1;
        int newK = (int) min(k, small);
        return newK;
    }

    public RealMatrix normal(double mean, double sd, int rows, int cols) {
        RealMatrix o = MatrixUtils.createRealMatrix(rows, cols);
        NormalDistribution normalDistribution = new NormalDistribution(mean, sd);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                o.setEntry(i, j, normalDistribution.sample());
            }
        }
        return o;
    }


    public void setWordScoreCalc(WordScoreCalc wordScoreCalc) {
        this.wordScoreCalc = wordScoreCalc;
    }

    @FunctionalInterface
    public static interface WordScoreCalc {
        double score(String word, int wordCountInDoc, int allWordCountInDoc);
    }

    public static interface Dict {
        Integer getIndex(String word);

        Integer size();
    }

    public void setDict(Dict dict) {
        this.dict = dict;
    }
}
