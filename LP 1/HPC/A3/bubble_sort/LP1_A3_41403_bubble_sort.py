# -*- coding: utf-8 -*-
"""HPC_3

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rTTnoC7mNIlAteHuYVblvmofoxhhV2ZE
"""

bubble_sort_2_threads = """
#include<omp.h>
#include<bits/stdc++.h>
using namespace std;

void swap(int *num1, int *num2) {
    int temp = *num1;
    *num1 = *num2;
    *num2 = temp;
}

int main() {
    int n = 10;
    int a[n];
    
    omp_set_num_threads(2);
    
    for(int i=0; i<n; i++) {
        a[i] = rand()% 100;
    }
    
    for(int i=0; i<n; i++) 
        cout<<a[i]<<" ";
    cout<<endl;
    
    int i=0, j=0;
    int first=0;
    double start, end;
    
    start = omp_get_wtime();
    for(i=0; i<n-1; i++) {
        first = i%2;
        #pragma omp parallel for
        for(j=first; j<n-1; j++) {
            if(a[j] > a[j+1])
              swap(&a[j], &a[j+1]);
        }
    }
    
    end = omp_get_wtime();
    cout<<"Result(parallel) : "<<endl;
    for(i=0; i<n; i++)
      cout<<a[i]<<" ";
    cout<<endl;
    
    cout<<"Time parallel = "<<(end-start)<<endl;
    
    return 0;
    
}
"""

bubble_sort_2_threads_code = open("bubblesort_2_thread.cpp", "w")
bubble_sort_2_threads_code.write(bubble_sort_2_threads)
bubble_sort_2_threads_code.close()

"""## 2 Threads"""

!g++ -fopenmp bubblesort_2_thread.cpp
!./a.out
