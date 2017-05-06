package com.github.reinvent.the.wheel.lsi;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.Collector;
import java.util.stream.Collectors;

public class LsiTest {
    public static void main(String[] args) throws IOException {
        Lsi lsi = new Lsi();
        Map<String, Integer> word2Index = new HashMap<>();
        word2Index.put("a", 0);
        word2Index.put("is", 1);
        word2Index.put("dog", 2);
        word2Index.put("cat", 3);
        word2Index.put("not", 4);
        word2Index.put("book", 5);
        word2Index.put("open", 6);
        word2Index.put("and", 7);
        word2Index.put("read", 8);
        word2Index.put("about", 9);
        Lsi.Dict dict = new Lsi.Dict() {
            @Override
            public Integer getIndex(String word) {
                return word2Index.get(word);
            }

            @Override
            public Integer size() {
                return word2Index.size();
            }
        };
        lsi.setDict(dict);

        String file = "/home/shaoaq/scm/github/reinvent-the-wheel/lsi/lsi-py/data2.txt";
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            List<String[]> documents = new ArrayList<>();
            reader.lines().forEach(line -> {
                documents.add(line.split(" "));
                if (documents.size() >= 3) {
                    lsi.addDocuments(documents);
                    documents.clear();
                }
            });
            lsi.addDocuments(documents);
        }
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            List<String[]> documents = new ArrayList<>();
            reader.lines().forEach(line -> {
                documents.add(line.split(" "));
            });
            RealMatrix vector = lsi.getVector(documents);
            System.out.println(vector);
            Map<String, RealVector> a = new HashMap<>();
            for (int i = 0; i < documents.size(); i++) {
                a.put(Arrays.stream(documents.get(i)).collect(Collectors.joining(" ")), vector.getRowVector(i));
            }
            String key ="buy a book about a cat";
            System.out.println(key);

            RealVector realVector = lsi.getVector(key.split(" "));
            System.out.println(realVector);
            a.entrySet().stream().map(e -> {
                double distance = e.getValue().getDistance(realVector);
                return new AbstractMap.SimpleEntry<String, Double>(e.getKey(), distance);
            }).sorted((e1, e2) -> Double.compare(e1.getValue(), e2.getValue()))
                    .forEach(e -> {
                        System.out.println(e.getValue() + ":" + e.getKey());
                    });
            ;
        }

    }
}
