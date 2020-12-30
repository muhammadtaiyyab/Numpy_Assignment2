{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[ 1  2]\n",
      "  [ 3  4]\n",
      "  [ 5  6]]\n",
      "\n",
      " [[ 7  8]\n",
      "  [ 9 10]\n",
      "  [11 12]]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "arr = np.array([[1, 2], [3, 4],[5, 6],[7, 8],[9, 10],[11, 12]])\n",
    "\n",
    "x = arr.reshape(2, 3, -1)\n",
    "\n",
    "print(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0]], [[28.0, 29.0, 30.0], [31.0, 32.0, 33.0], [34.0, 35.0, 36.0]]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "myarray = ([[[10., 11., 12.],\n",
    "        [13., 14., 15.],\n",
    "        [16., 17., 18.]],\n",
    "       [[19., 20., 21.],\n",
    "        [22., 23., 24.],\n",
    "        [25., 26., 27.]],\n",
    "       [[28., 29., 30.],\n",
    "        [31., 32., 33.],\n",
    "        [34., 35., 36.]]]) \n",
    "print(myarray)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 6, 7, 8])"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "T = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])\n",
    "\n",
    "T[1,:]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1 2]\n",
      " [3 4 5]\n",
      " [6 7 8]]\n",
      "[[1 0 2]\n",
      " [4 3 5]\n",
      " [7 6 8]]\n",
      "[[6 7 8]\n",
      " [3 4 5]\n",
      " [0 1 2]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "my_array = np.arange(9).reshape(3, 3)\n",
    "print( my_array)\n",
    "def swap_cols(arr, frm, to):\n",
    "    arr[:,[frm, to]] = arr[:,[to, frm]]\n",
    "\n",
    "swap_cols(my_array, 0, 1)\n",
    "print (my_array)\n",
    "def swap_rows(arr, frm, to):\n",
    "    arr[[frm, to],:] = arr[[to, frm],:]\n",
    "\n",
    "my_array = np.arange(9).reshape(3, 3)\n",
    "swap_rows(my_array, 0, 2)\n",
    "print (my_array)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      "Update sixth value to 11\n",
      "[ 0.  0.  0.  0.  0.  0. 11.  0.  0.  0.]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "x = np.zeros(10)\n",
    "print(x)\n",
    "print(\"Update sixth value to 11\")\n",
    "x[6] = 11\n",
    "print(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 6"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      "Update sixth value to 10\n",
      "[ 0.  0.  0.  0.  0.  0. 10.  0.  0.  0.]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "x = np.zeros(10)\n",
    "print(x)\n",
    "print(\"Update sixth value to 10\")\n",
    "x[6] = 10\n",
    "print(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 7"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0. 0. 0. 0. 0. 0. 0. 0.]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "a = np.zeros(8)\n",
    "print(a)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 8"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[6 6 6 6 6]\n",
      " [6 6 6 6 6]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "#using no.full\n",
    "x = np.full((2, 5), 6, dtype=np.uint)\n",
    "print(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 9"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23\n",
      " 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47\n",
      " 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71\n",
      " 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95\n",
      " 96 97 98 99]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "array = np.arange(100)\n",
    "print (array)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1st Input array :  [[3 3 3]\n",
      " [4 4 4]\n",
      " [5 5 5]]\n",
      "2nd Input array :  [1 2 3]\n",
      "Output array:  [[2 1 0]\n",
      " [3 2 1]\n",
      " [4 3 2]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as geek \n",
    "  \n",
    "in_arr = np.array([[3,3,3],[4,4,4],[5,5,5]])\n",
    "in_brr = np.array([1,2,3])\n",
    "   \n",
    "print (\"1st Input array : \", in_arr) \n",
    "print (\"2nd Input array : \", in_brr) \n",
    "   \n",
    "    \n",
    "out_arr1 = geek.subtract(in_arr, in_brr)  \n",
    "print (\"Output array: \", out_arr1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 11"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[-1  2 -1  4 -1  6 -1  8 -1 10]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np \n",
    " \n",
    "a = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]) \n",
    " \n",
    "odd_values = (a%2 == 1) \n",
    "a[odd_values] = -1\n",
    "print (a)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 12"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1, 3, 8]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "values = np.array([0,1,2,1,2,4,5,6,1,2,1])\n",
    "searchval = [1,2]\n",
    "N = len(searchval)\n",
    "possibles = np.where(values == searchval[0])[0]\n",
    "\n",
    "solns = []\n",
    "for p in possibles:\n",
    "    check = values[p:p+N]\n",
    "    if np.all(check == searchval):\n",
    "        solns.append(p)\n",
    "\n",
    "print(solns)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 13"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "a: [[1 3 2 4]\n",
      " [0 3 3 2]\n",
      " [1 2 3 4]\n",
      " [4 4 4 0]\n",
      " [0 4 3 0]]\n",
      "\n",
      "b: [[0 1 0 1]\n",
      " [0 1 1 0]\n",
      " [0 0 1 1]\n",
      " [1 1 1 0]\n",
      " [0 1 1 0]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "a = np.random.randint(0, 5, size=(5, 4))\n",
    "b = np.where(a<3,0,1)\n",
    "print('a:',a)\n",
    "print()\n",
    "print('b:',b)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 14"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[array([10, 11, 12, 13, 14, 15]), array([16, 17, 18, 19, 20, 21]), array([22, 23, 24, 25, 26, 27]), array([28, 29, 30, 31, 32, 33])]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "arr = np.arange(10, 34, 1)\n",
    "\n",
    "newarr = np.array_split(arr, 4)\n",
    "\n",
    "print(newarr)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 15"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-2  2  8]\n",
      " [-4  1  7]\n",
      " [ 3  6  9]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "arr = np.array([[ 8,  2, -2],[-4,  1,  7],[ 6,  3,  9]])\n",
    "\n",
    "print(np.sort(arr))\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 16"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1 2]\n",
      " [2 3]\n",
      " [3 4]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "x = np.array([[1], [2], [3]])\n",
    "y = np.array([[2], [3], [4]])\n",
    "\n",
    "arr = np.concatenate((x, y), axis=1)\n",
    "\n",
    "print(arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 17"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "NO\n",
      "NO\n",
      "WHATEVER\n",
      "YES\n",
      "YES\n",
      "YES\n",
      "YES\n"
     ]
    }
   ],
   "source": [
    "for i in range(-2, 5):\n",
    "  if i > 0:\n",
    "    print(\"YES\")\n",
    "  elif i == 0:\n",
    "    print(\"WHATEVER\")\n",
    "  else:\n",
    "    print(\"NO\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 18"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2\n"
     ]
    }
   ],
   "source": [
    "points = [1, 4, 2, 9, 7, 8, 9, 3, 1]\n",
    "\n",
    "x = points.count(9)\n",
    "print (x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 19"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 2 1 2 3 4]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "arr = np.array([-1, -2, 1, 2, 3, -4])\n",
    "\n",
    "newarr = np.absolute(arr)\n",
    "\n",
    "print(newarr)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Task no 20"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1 2]\n",
      " [3 4]\n",
      " [5 6]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np \n",
    "a = np.array([[1,2,3],[4,5,6]]) \n",
    "b = a.reshape(3,2) \n",
    "print (b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
