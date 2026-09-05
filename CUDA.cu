__global__ void test_kernel() {}

int main() {
    test_kernel<<<1, 100000>>>();
    return 0;
}