"use strict";(self.webpackChunknotes_template=self.webpackChunknotes_template||[]).push([[368],{4966:(e,i,n)=>{n.r(i),n.d(i,{assets:()=>a,contentTitle:()=>t,default:()=>h,frontMatter:()=>o,metadata:()=>l,toc:()=>c});const l=JSON.parse('{"id":"sycl/basic-kernel","title":"Basic Kernel","description":"A kernel is a task that is distributed to each processing element on a computation device.","source":"@site/docs/sycl/basic-kernel.mdx","sourceDirName":"sycl","slug":"/sycl/basic-kernel","permalink":"/parallel-programming-essentials/docs/sycl/basic-kernel","draft":false,"unlisted":false,"editUrl":"https://github.com/finger-bone/parallel-programming-essentials/blob/main/docs/sycl/basic-kernel.mdx","tags":[],"version":"current","sidebarPosition":3,"frontMatter":{"sidebar_position":3},"sidebar":"tutorialSidebar","previous":{"title":"Memory","permalink":"/parallel-programming-essentials/docs/sycl/memory"},"next":{"title":"Parallel Prefix Sum","permalink":"/parallel-programming-essentials/docs/parallel-prefix-sum"}}');var r=n(4848),s=n(8453);const o={sidebar_position:3},t="Basic Kernel",a={},c=[{value:"Expressing Parallelism",id:"expressing-parallelism",level:2},{value:"Multidimensional <code>parallel_for</code>",id:"multidimensional-parallel_for",level:2}];function d(e){const i={admonition:"admonition",code:"code",em:"em",h1:"h1",h2:"h2",header:"header",p:"p",pre:"pre",...(0,s.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(i.header,{children:(0,r.jsx)(i.h1,{id:"basic-kernel",children:"Basic Kernel"})}),"\n",(0,r.jsx)(i.p,{children:"A kernel is a task that is distributed to each processing element on a computation device."}),"\n",(0,r.jsxs)(i.admonition,{type:"note",children:[(0,r.jsx)(i.p,{children:"In SYCL, there are two types of kernel."}),(0,r.jsx)(i.p,{children:"The basic kernel, which we are introducing here, is a type that is simple and more managed by the run time. It is should be your goto-choice, because it allows you to focus on the logic."}),(0,r.jsx)(i.p,{children:"However, there are also ND kernel which allows more control for you, and this is more similar to the kernel you'll use in CUDA or OpenCL. This is more traditional and if you were migrating a kernel from CUDA or OpenCL, you should use the ND kernel."}),(0,r.jsx)(i.p,{children:"Again, this series choose SYCL for its simplicity, so we only introduce the basic kernel. Unless you need extreme performance, or you're optimizing the hot spot, you should use the basic kernel for more productivity."})]}),"\n",(0,r.jsx)(i.h2,{id:"expressing-parallelism",children:"Expressing Parallelism"}),"\n",(0,r.jsxs)(i.p,{children:["We used ",(0,r.jsx)(i.code,{children:"parallel_for"})," before- but we never explained how it wo=rks. We just told you what action we achieved. Now let's look at how to use it."]}),"\n",(0,r.jsxs)(i.p,{children:["Serial programming ",(0,r.jsx)(i.code,{children:"for"})," would look like,"]}),"\n",(0,r.jsx)(i.pre,{children:(0,r.jsx)(i.code,{className:"language-cpp",children:"for (int i = 0; i < N; i++) {\n    a[i] = b[i] + c[i];\n}\n"})}),"\n",(0,r.jsxs)(i.p,{children:["This loop will first, set an ",(0,r.jsx)(i.code,{children:"i"})," value, execute the body, then set a new value or exit the loop."]}),"\n",(0,r.jsx)(i.p,{children:"But for parallel for, we use the following,"}),"\n",(0,r.jsx)(i.pre,{children:(0,r.jsx)(i.code,{className:"language-cpp",children:"q.submit([&](handler &h) {\n    h.parallel_for(N, [=](id<1> i) {\n        a[i] = b[i] + c[i];\n    })\n})\n"})}),"\n",(0,r.jsxs)(i.p,{children:["The first parameter of ",(0,r.jsx)(i.code,{children:"parallel_for"})," is ",(0,r.jsx)(i.em,{children:"how many kernels you'd like to launch"}),". The second parameter is the kernel function."]}),"\n",(0,r.jsx)(i.p,{children:"When executing the command, the SYCL runtime will distribute the kernel function to each processing element (the minimal unit of execution on a device)."}),"\n",(0,r.jsxs)(i.p,{children:["However, we definitely do not want all of the kernel function to repeat the exact same task. So besides the kernel function, every processing element will also get their ",(0,r.jsx)(i.code,{children:"id"}),"."]}),"\n",(0,r.jsxs)(i.p,{children:["For the above code, a processing element will add up the ",(0,r.jsx)(i.code,{children:"id"}),"-th dimension of the two vector."]}),"\n",(0,r.jsx)(i.p,{children:"Each of such a kernel is also called a work item."}),"\n",(0,r.jsx)(i.p,{children:"Again, each kernel runs parallel. In parallel programming, the work of us, the programmer, is to design the kernel function, so that they can run parallel without blocking (or as little blocking as possible), and be as fast as possible."}),"\n",(0,r.jsxs)(i.h2,{id:"multidimensional-parallel_for",children:["Multidimensional ",(0,r.jsx)(i.code,{children:"parallel_for"})]}),"\n",(0,r.jsxs)(i.p,{children:["We may sometimes write double loop for matrix operation. A simple way to write kernel for such operation would be to flatten the matrix to one dimensional array. But since this is so frequently used, multidimensional ",(0,r.jsx)(i.code,{children:"parallel_for"})," is provided."]}),"\n",(0,r.jsx)(i.pre,{children:(0,r.jsx)(i.code,{className:"language-cpp",children:"h.parallel_for(range{N, N}, =[=](id<2> idx) {\n    int j = idx[0];\n    int i = idx[1];\n    for (int k = 0; k < N; ++k) {\n        c[j][i] += a[j][k] * b[k][i]; // or c[idx] += a[id(j,k)] * b[id(k,i)];\n    }\n});\n"})}),"\n",(0,r.jsx)(i.p,{children:"Please take note that the maximum number of dimension is three."})]})}function h(e={}){const{wrapper:i}={...(0,s.R)(),...e.components};return i?(0,r.jsx)(i,{...e,children:(0,r.jsx)(d,{...e})}):d(e)}}}]);