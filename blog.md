<h1 align="center"><b>Softmax Regression</b></h1>

## GIỚI THIỆU
Hồi quy Softmax (Softmax Regression) là một thuật toán học có giám sát (supervised learning), mặc dù tên gọi có chứa từ "hồi quy" nhưng đây là thuật toán thuộc loại classification. Nó tính toán mối quan hệ giữa các đặc trưng trong input và output dựa trên hàm softmax. Thực tế cho thấy nó là một trong những thuật toán Machine Learning được sử dụng phổ biến nhất.

Hồi quy Softmax (hay hồi quy logistic đa thức) là tổng quát của hồi quy logistic trong trường hợp chúng ta muốn xử lý nhiều lớp. Trong hồi quy logistic, chung tôi giả định rằng các nhãn là nhị phân <img src="https://render.githubusercontent.com/render/math?math=y^{i} \in \{0, 1\}">, nhưng trong hồi quy Softmax cho phép chúng tôi xử lý <img src="https://render.githubusercontent.com/render/math?math=y^{i} \in \{1, ..., K\}"> với <img src="https://render.githubusercontent.com/render/math?math=K"> là số lớp.

Trong cài đặt hồi quy softmax, chúng tôi quan tâm đến phân loại nhiều lớp (thay vì chỉ phân loại nhị phân), và vì vậy nhãn y có thể đảm nhiệm <img src="https://render.githubusercontent.com/render/math?math=K"> các giá trị khác nhau, thay vì chỉ có hai. 

## PHƯƠNG PHÁP
Ý tưởng của bài toán là 

## THAM KHẢO
1. http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/
