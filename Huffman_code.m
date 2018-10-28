function [ h, e ] = Huffman_code( p )

if ~isempty(find(p<0, 1))
    error('the probablity can not be less than 0')
end
 
if abs(sum(p)-1)>10e-10
    error('the whole probablity can not be more than 1')
end
 
 
n=length(p);
 
p=sort(p);
q=p;
m=zeros(n-1,n);
for i=1:n-1
    [q,e]=sort(q); 
    m(i,:)=[e(1:n-i+1),zeros(1,i-1)]; %ÓÉÊý×él ¹¹½¨Ò»¸ö¾ØÕó£¬¸Ã¾ØÕó±íÃ÷¸ÅÂÊºÏ²¢Ê±µÄË³Ðò£¬ÓÃÓÚºóÃæµÄ±àÂë
    q=[q(1)+q(2),q(3:n),1]; 
end
 
for i=1:n-1
    c(i,1:n*n)=blanks(n*n); %c ¾ØÕóÓÃÓÚ½øÐÐhuffman ±àÂë
end
    c(n-1,n)='1'; %ÓÉÓÚa ¾ØÕóµÄµÚn-1 ÐÐµÄÇ°Á½¸öÔªËØÎª½øÐÐhuffman ±àÂë¼ÓºÍÔËËãÊ±ËùµÃµÄ×îºóÁ½¸ö¸ÅÂÊ(ÔÚ±¾ÀýÖÐÎª0.02¡¢0.08)£¬Òò´ËÆäÖµÎª0 »ò1
    c(n-1,2*n)='0'; 
for i=2:n-1
    c(n-i,1:n-1)=c(n-i+1,n*(find(m(n-i+1,:)==1))-(n-2):n*(find(m(n-i+1,:)==1))); %¾ØÕóc µÄµÚn-i µÄµÚÒ»¸öÔªËØµÄn-1 µÄ×Ö·û¸³ÖµÎª¶ÔÓ¦ÓÚa ¾ØÕóÖÐµÚn-i+1 ÐÐÖÐÖµÎª1 µÄÎ»ÖÃÔÚc ¾ØÕóÖÐµÄ±àÂëÖµ
    c(n-i,n)='0'; 
    c(n-i,n+1:2*n-1)=c(n-i,1:n-1); %¾ØÕóc µÄµÚn-i µÄµÚ¶þ¸öÔªËØµÄn-1 µÄ×Ö·ûÓëµÚn-i ÐÐµÄµÚÒ»¸öÔªËØµÄÇ°n-1 ¸ö·ûºÅÏàÍ¬£¬ÒòÎªÆä¸ù½ÚµãÏàÍ¬
    c(n-i,2*n)='1'; 
    for j=1:i-1
         c(n-i,(j+1)*n+1:(j+2)*n)=c(n-i+1,n*(find(m(n-i+1,:)==j+1)-1)+1:n*find(m(n-i+1,:)==j+1));
            %¾ØÕóc ÖÐµÚn-i ÐÐµÚj+1 ÁÐµÄÖµµÈÓÚ¶ÔÓ¦ÓÚa ¾ØÕóÖÐµÚn-i+1 ÐÐÖÐÖµÎªj+1 µÄÇ°ÃæÒ»¸öÔªËØµÄÎ»ÖÃÔÚc ¾ØÕóÖÐµÄ±àÂëÖµ
    end
end 
for i=1:n
    h(i,1:n)=c(1,n*(find(m(1,:)==i)-1)+1:find(m(1,:)==i)*n); %ÓÃh±íÊ¾×îºóµÄhuffman ±àÂë
    len(i)=length(find(abs(h(i,:))~=32)); %¼ÆËãÃ¿Ò»¸ö±àÂëµÄ³¤¶È
end
e=sum(p.*len); %¼ÆËãÆ½¾ùÂë³¤
