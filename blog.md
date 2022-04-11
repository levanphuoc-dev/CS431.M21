<h1 align="center"><b>Softmax Regression</b></h1>

## GIỚI THIỆU
Hồi quy Softmax (Softmax Regression) là một thuật toán học có giám sát (supervised learning), mặc dù tên gọi có chứa từ "hồi quy" nhưng đây là thuật toán thuộc loại classification. Nó tính toán mối quan hệ giữa các đặc trưng trong input và output dựa trên hàm softmax. Thực tế cho thấy nó là một trong những thuật toán Machine Learning được sử dụng phổ biến nhất.

Hồi quy Softmax (hay hồi quy logistic đa thức) là tổng quát của hồi quy logistic trong trường hợp chúng ta muốn xử lý nhiều lớp. Trong hồi quy logistic, chung tôi giả định rằng các nhãn là nhị phân <img src="https://render.githubusercontent.com/render/math?math=y^{i} \in \{0, 1\}">, nhưng trong hồi quy Softmax cho phép chúng tôi xử lý <img src="https://render.githubusercontent.com/render/math?math=y^{i} \in \{1, ..., K\}"> với <img src="https://render.githubusercontent.com/render/math?math=K"> là số lớp.

Trong cài đặt hồi quy softmax, chúng tôi quan tâm đến phân loại nhiều lớp (thay vì chỉ phân loại nhị phân), và vì vậy nhãn y có thể đảm nhiệm <img src="https://render.githubusercontent.com/render/math?math=K"> các giá trị khác nhau, thay vì chỉ có hai. 

## PHƯƠNG PHÁP
Ý tưởng của bài toán là tương tự như bài toán hồi quy logistic, bài toán hồi quy softmax thay thế hàm sigmoid thành hàm softmax để có thể sử dụng cho bài toán phân loại nhiều lớp hơn.

Chúng ta cần một mô hình xác suất sao cho với mỗi input <img src="https://render.githubusercontent.com/render/math?math=x">, <img src="https://render.githubusercontent.com/render/math?math=a_i"> thể hiện xác suất để input đó rơi vào lớp i. Vậy điều kiện cần là các <img src="https://render.githubusercontent.com/render/math?math=a_i"> phải dương và tổng của chúng bằng 1. Để có thể thỏa mãn điều kiện này, chung ta cần nhìn vào mọi giá trị <img src="https://render.githubusercontent.com/render/math?math=z_i"> và dựa trên các quan hệ giữa các <img src="https://render.githubusercontent.com/render/math?math=z_i"> này để tính toán giá trị của <img src="https://render.githubusercontent.com/render/math?math=a_i">.
Ngoài các điều kiện <img src="https://render.githubusercontent.com/render/math?math=a_i"> lớn hơn 0 và có tổng bằng 1, chúng ta sẽ thêm một điều kiện cũng rất tự nhiên nữa, đó là: giá trị <img src="https://render.githubusercontent.com/render/math?math=z_i = \theta_i^T x"> càng lớn thì xác suất dữ liệu rơi vào lớp i càng cao.
Điều kiện cuối này chỉ ra rằng chúng ta cần một hàm đồng biến ở đây.

Chú ý rằng <img src="https://render.githubusercontent.com/render/math?math=z_i"> có thể nhận giá trị cả âm và dương. Vì thế ta sử dụng hàm <img src="https://render.githubusercontent.com/render/math?math=exp(z_i) = e^{z_i}"> thì có thể chắc chắn biến <img src="https://render.githubusercontent.com/render/math?math=z_i"> thành một số dương, đồng biến. Điều kiện cuối cùng, tổng các <img src="https://render.githubusercontent.com/render/math?math=a_i"> bằng 1 có thể được đảm bảo nếu:

<img src="https://render.githubusercontent.com/render/math?math=a_i = \frac{exp(z_i)}{\sum_{i=1}^C exp(z_j)},  \forall_i = 1, 2, ..., C"> 

Hàm số này, tính tất cả các <img src="https://render.githubusercontent.com/render/math?math=a_i"> dựa vào tất cả các <img src="https://render.githubusercontent.com/render/math?math=z_i">, thỏa mãn tất cả các điều kiện đã xét: dương, tổng bằng 1, giữ được thứ tự của <img src="https://render.githubusercontent.com/render/math?math=z_i">. Hàm số này được gọi là hàm softmax.

Lúc này, ta có thể giả sử rằng:

<img src="https://render.githubusercontent.com/render/math?math=P(y_k = i \mid x_k">; <img src="https://render.githubusercontent.com/render/math?math=\theta) = a_i">

Trong đó, <img src="https://render.githubusercontent.com/render/math?math=P(y_k = i \mid x_k">; <img src="https://render.githubusercontent.com/render/math?math=\theta)"> được hiểu là xác suất để một điểm dữ liệu <img src="https://render.githubusercontent.com/render/math?math=x"> rơi vào lớp thứ i nếu biết tham số mô hình (ma trận trọng số) là <img src="https://render.githubusercontent.com/render/math?math=\theta">.

## THAM KHẢO
1. http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/
