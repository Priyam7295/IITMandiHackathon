import { motion } from 'framer-motion';

const languages = [
  { name: 'Bengali', code: 'bn' },
  { name: 'Gujarati', code: 'gu' },
  { name: 'Hindi', code: 'hi' },
  { name: 'Kannada', code: 'kn' },
  { name: 'Malayalam', code: 'ml' },
  { name: 'Marathi', code: 'mr' },
  { name: 'Punjabi', code: 'pa' },
  { name: 'Tamil', code: 'ta' },
  { name: 'Telugu', code: 'te' },
  { name: 'Urdu', code: 'ur' }
];

export default function LanguagesSection() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        when: "beforeChildren"
      }
    }
  };

  const itemVariants = {
    hidden: { y: 30, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.6,
        ease: [0.25, 0.1, 0.25, 1]
      }
    },
    hover: {
      y: -5,
      scale: 1.05,
      boxShadow: "0 10px 25px -5px rgba(192, 132, 252, 0.2)",
      transition: {
        duration: 0.3,
        ease: "easeOut"
      }
    }
  };

  const titleVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.8,
        ease: [0.25, 0.1, 0.25, 1]
      }
    }
  };

  return (
    <motion.section
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: "-100px" }}
      variants={containerVariants}
      className="bg-white/5 backdrop-blur-sm p-8 rounded-xl border border-white/10 shadow-lg relative overflow-hidden"
    >
      {/* Animated background elements */}
      <motion.div 
        className="absolute -top-20 -left-20 w-40 h-40 bg-purple-500/10 rounded-full filter blur-xl"
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.1, 0.2, 0.1]
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      <motion.div 
        className="absolute -bottom-20 -right-20 w-60 h-60 bg-emerald-500/10 rounded-full filter blur-xl"
        animate={{
          scale: [1, 1.3, 1],
          opacity: [0.1, 0.15, 0.1]
        }}
        transition={{
          duration: 12,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 2
        }}
      />
      
      <motion.h2 
        variants={titleVariants}
        className="text-3xl font-bold mb-8 text-center text-purple-300 relative z-10"
      >
        Supported Indian Languages
        <motion.div 
          className="w-24 h-1 bg-gradient-to-r from-purple-500 to-pink-500 mx-auto mt-2 rounded-full"
          initial={{ width: 0 }}
          whileInView={{ width: "6rem" }}
          viewport={{ once: true }}
          transition={{ delay: 0.3, duration: 0.8 }}
        />
      </motion.h2>
      
      <motion.div 
        className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4 relative z-10"
        variants={containerVariants}
      >
        {languages.map((language, index) => (
          <motion.div
            key={language.code}
            variants={itemVariants}
            whileHover="hover"
            className="p-4 bg-white/5 rounded-lg border border-white/10 text-center cursor-default"
            custom={index}
          >
            <motion.span 
              className="text-lg font-medium text-white"
              whileHover={{ color: "#e879f9" }}
            >
              {language.name}
            </motion.span>
            <motion.div 
              className="text-xs text-purple-300 mt-1"
              initial={{ opacity: 0 }}
              whileHover={{ opacity: 1 }}
            >
              {language.code}
            </motion.div>
          </motion.div>
        ))}
      </motion.div>
      
      {/* Floating decorative elements */}
      <motion.div 
        className="absolute top-1/4 left-1/4 w-2 h-2 bg-purple-400 rounded-full"
        animate={{
          y: [0, -15, 0],
          opacity: [0.6, 1, 0.6]
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      <motion.div 
        className="absolute top-3/4 right-1/3 w-3 h-3 bg-emerald-400 rounded-full"
        animate={{
          y: [0, 15, 0],
          opacity: [0.6, 1, 0.6]
        }}
        transition={{
          duration: 5,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 1
        }}
      />
    </motion.section>
  );
}