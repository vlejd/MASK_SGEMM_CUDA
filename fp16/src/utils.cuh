/*
#pragma once
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>

#define cudaCheck(err) (cudaCheckInternal(err, __FILE__, __LINE__))


void log_matrix_data(const std::string &fileName, const Problem_InstanceFP32 &pi) {
    std::ofstream fs;
    fs.open(fileName);
    fs << "A:\n";
    print_matrix(pi.hA, pi.M, pi.K, fs);
    fs << "B:\n";
    print_matrix(pi.hB, pi.K, pi.N, fs);
    fs << "Bt:\n";
    print_matrix(pi.hBt, pi.K, pi.N, fs);
    fs << "Mask:\n";
    print_matrix(pi.hMask, pi.K, pi.N, fs);
    fs << "C:\n";
    print_matrix(pi.hC, pi.M, pi.N, fs);
    fs << "Should:\n";
    print_matrix(pi.hC_ref, pi.M, pi.N, fs);
};
*/