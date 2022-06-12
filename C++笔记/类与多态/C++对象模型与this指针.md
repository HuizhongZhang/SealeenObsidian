#### 1 C++对象模型
##### 1.1 成员变量和成员函数分开存储
只有非静态成员变量属于类的对象，函数（所有对象共用一个函数实例）和静态成员变量不属于（空对象占一字节）；
```cpp
class Person {
public:
    // 静态成员函数，不占对象空间
    static void func1();
    // 静态成员变量，不占对象空间
    static int s_a_;
    // 非静态成员函数，所有对象共用一份实例，不占对象空间
    void func2();
private:
    // 非静态成员变量，占对象空间
    int b_;
};
// 静态成员变量一定要类外初始化
int Person::s_a_ = 10;

void test01() {
    Person p; // 空对象占一个字节，只有非静态变量占对象内存
    cout << "sizeof(p): " << sizeof(p) << endl;
}
```
stdout:
```cpp
sizeof(p): 4
```

#### 2 This指针
- **定义**：指向调用成员函数对象的指针（指针常量，指向不能修改），隐含在每一个非静态成员函数里，可直接使用；
![[Pasted image 20220612095226.png]]
- **作用**：（0）所有对象共用一份非静态成员函数，系统通过```this```指针来确定函数作用的对象;
             （1）形参和成员变量同名时，可解决命名冲突```this->val```;
             （2）在非静态成员函数中返回对象本身```return *this```;
```cpp
class Person {
public:
    Person(int age) {
        // age = age; // 编译器认为age都是形参变量
        this->age = age;
    }
    // 返回对象本身
    Person PersonAddAge(const Person &p) {
        this->age += p.age;
        return *this;
    }
    int age;
};
// 1.成员变量与形参同名时，可用于解决命名冲突
void test01() {
    Person p(20);
    cout << "年龄：" << p.age << endl;
}
// 2.非静态成员函数返回对象本身
void test02() {
    Person p1(20);
    Person p2(10);
    // 返回对象后，可实现链式编程
    p2 = p2.PersonAddAge(p1).PersonAddAge(p1).PersonAddAge(p1);
    cout << "年龄：" << p2.age << endl;
}
```
stdout:
```cpp
年龄：20
年龄：70
```

#### 3 空指针调用成员函数
- **作用**：空对象指针可以调成员函数，但不能包含成员变量（this指针为空报错）；
```cpp
class Person {
public:
    Person(int age) {
        age_ = age;
    }
    void showClassName() {
        cout << "class name is Person" << endl;
    }
    void showAge() {
        // 提高代码健壮性
        if (this == nullptr) {
            cout << "this指针为空" << endl;
            return;
        }
        cout << "age is" << age_ << endl; // 等同于this->age_
    }
    int age_;
};

// 空指针调用成员函数
void test01() {
    Person* p = nullptr;
    p->showClassName();
    p->showAge(); // 不能包含成员变量,this指针为空报错
}
```
stdout:
```cpp
class name is Person
this指针为空
```

#### 4 常函数和常对象
- **常函数**：成员函数后加const，不能修改成员属性（成员属性声明时加mutable除外）
- **常对象**：声明对象前加const，常对象只能调用常函数；
