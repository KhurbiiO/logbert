file="../output/custom"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi

file="../output/custom/bert"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi