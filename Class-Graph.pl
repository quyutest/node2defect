use Understand;

my $path = $ENV{'PATH'};
my @temp;
@paths = split(";",$path);
foreach $pa(@paths){
  if(($pa=~m/jdk/)!=1){
      push(@temp,$pa);
  }
}
$ENV{'PATH'} = join(";",@temp);
print $ENV{'PATH'};

system( "und -quiet create -db temp.udb -languages java add " . $ARGV[0] . " analyze" );
$project_name=$ARGV[0];

( $db, $status ) = Understand::open("temp.udb");
die "Error status: ", $status, "\n" if $status;
$output_file=">classgraph.dot";
open( OUTFILE, $output_file );
print OUTFILE "digraph G{\n";
my %hashmap;

$output_file2=">CountLine-".$project_name.".csv";
open(OUTFILE2, $output_file2);

sub printClassGraph{
	my $category = shift(@_);
	my $callee = shift(@_);
	my $caller = shift(@_);
	$callee_name=$callee->longname();
	$caller_name=$caller->longname();

	$Anon="Anon";
	if(($callee_name=~/$Anon/)||($caller_name=~/$Anon/)){
		last;
	}

	if ( !exists $hashmap{$caller_name} ) {
		$hashmap{$caller_name} = 1;
		print OUTFILE "\t" . "\"" . $caller_name . "\"" . ";" . "\n";
	}
	if ( !exists $hashmap{$callee_name} ) {
		$hashmap{$callee_name} = 1;
		print OUTFILE "\t" . "\"" . $callee_name . "\"" . ";" . "\n";
	}

	print OUTFILE "\t" . "\"" 
	. $caller_name . "\"" . "->" . "\"" 
	. $callee_name . "\""
	. "["
	. $category . "]"
	. ";" . "\n";
}

foreach $interface ($db->ents("Interface")){
	$loc=$interface->metric("CountLineCode");
	if($loc==0){
		next;
	}
	foreach $class ($interface->ents("Implementby","Class")){
		$loc_class=$class->metric("CountLineCode");
		if($loc_class==0){
			next;
		}
		printClassGraph("I",$interface,$class);#实例化接口关系用I表示。
	}
}

@abstract_class_list=$db->ents("Abstract Class");
@class_list=$db->ents("Class");
@interface_list=$db->ents("Interface");
@all_class_list=(@abstract_class_list,@class_list,@interface_list);

$all_class_length=@all_class_list;
my %count;
@all_class_unique_list=grep { ++$count{ $_ } < 2; } @all_class_list;
$all_class_unique_length=@all_class_unique_list;


my %class_name_list;
$class_count=0;

foreach $class (@all_class_unique_list){
	$loc=$class->metric("CountLineCode");
	if($loc==0){
		next;
	}
	$Anon="Anon";
	if($class->longname()=~/$Anon/){
		next;
	}
    $class_count=$class_count+1;

	foreach $child_class ($class->ents("Extendby","Class")){
		$loc_class=$child_class->metric("CountLineCode");
		if($loc_class==0){
			next;
		}
		printClassGraph("C",$class,$child_class);#继承关系用C表示，代表Child的意思。
	}
	foreach $aggre_class ($class->ents("Coupleby","Class")){
		$loc_class=$aggre_class->metric("CountLineCode");
		if($loc_class==0){
			next;
		}
		printClassGraph("A",$class,$aggre_class);#聚合关系用A表示，代表Aggregation的意思。
	}
	foreach $method ($class->ents('Define','Method')){
		$loc_method=$method->metric("CountLineCode");
		if($loc_method==0){
			print "This method's loc==0:",$method->longname(),"\n";
			next;
		}
	}
	$class_name=$class->name();
	@names=split(/\./,$class_name);
	$class_name_list{$names[@names-1]}=$class;
}

foreach $class (@all_class_unique_list){
	$loc=$class->metric("CountLineCode");
	if($loc==0){
		next;
	}
	foreach $method ($class->ents('Define','Method')){
		$loc_method=$method->metric("CountLineCode");
		if($loc_method==0){
			print "This method's loc==0:",$method->longname(),"\n";
			next;
		}
		$method_return_type=$method->type();
		if(exists $class_name_list{$method_return_type}){
			printClassGraph("R",$class_name_list{$method_return_type},$class);#方法返回值用R表示，代表Return的意思。
		}
		$method_parameters=$method->parameters();
		@parameter_list=split(/\,/,$method_parameters);
		@parameter_type_list=();
		foreach $parameter (@parameter_list){
			@parameter_one=split(/\ /,$parameter);
			@parameter_one_strip=split(/\[/,@parameter_one[0]);
			@parameter_type_list=(@parameter_type_list,@parameter_one_strip[0]);
		}
		foreach $parameter_type (@parameter_type_list){
			if(exists $class_name_list{$parameter_type}){
				printClassGraph("P",$class_name_list{$parameter_type},$class);#参数类型用P表示，代表Parameter的意思。
				last;
			}
		}
	}
}

print OUTFILE "}";
print OUTFILE2 $db->metric("CountLineCode"),",";
print OUTFILE2 $class_count;

