# -*- coding: utf-8 -*-
"""HPC_3

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rTTnoC7mNIlAteHuYVblvmofoxhhV2ZE
"""

merge_sort_2_threads = """
#include<bits/stdc++.h>
#include<omp.h>

using namespace std;

void print(int *arr, int size) {
    for(int i=0; i<size; i++) {
         cout<<arr[i]<<"  ";
    }
    cout<<endl;
}

void merge(int* arr, int start, int mid, int end) {
    int len = (end - start) + 1;
    int temp[len];
    int cur = 0;
    
    int i = start;
    int j = mid + 1;
    while(i <= mid && j <= end){
        if(arr[i] < arr[j]) {
            temp[cur] = arr[i];
            cur++;
            i++;
        }
        else {
            temp[cur] = arr[j];
            cur++;
            j++;
        }
    }
    if(i <= mid) {
        while(i <= mid) {
            temp[cur] = arr[i];
            i++;
            cur++;
        }
    }
    
    else if(j <= end) {
        while(j <= end) {
            temp[cur] = arr[j];
            j++;
            cur++;
        }
    }
    
    cur = 0;
    for(i=start; i<=end; i++) {
        arr[i] = temp[cur];
        cur++;
    }

}

void mergeSort(int *arr, int start, int end) {
    if(start < end) {
        int mid = (start+end) / 2;
        
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSort(arr, start, mid);
            
            #pragma omp section
            mergeSort(arr, mid+1, end);
        }
        
        merge(arr, start, mid, end);
    
    }
}

int main(int argc, char *argv[]) {
    int size = 10;
    int a[size];
    
    double start, end;
    
    omp_set_num_threads(2);
    
    for(int i=0; i<size; i++) {
        a[i] = rand()% 100;
    }
    
    for(int i=0; i<size; i++) 
        cout<<a[i]<<" ";
    cout<<endl;
    start = omp_get_wtime();
    
    mergeSort(a, 0, size-1);
    
    print(a, size);
    
    end = omp_get_wtime();
    
    cout<<"Time = "<<(end-start)<<endl;
    
    return 0;
}

"""

merge_sort_2_threads_code = open("mergesort_2_thread.cpp", "w")
merge_sort_2_threads_code.write(merge_sort_2_threads)
merge_sort_2_threads_code.close()

"""## 2 Threads"""

!g++ -fopenmp mergesort_2_thread.cpp
!./a.out

