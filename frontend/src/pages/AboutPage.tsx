const AboutPage = () => {
  return (
    <div className="max-w-4xl mx-auto p-8">
      <h1 className="text-4xl font-bold mb-6">About ShikshaSetu</h1>
      <p className="text-lg text-gray-700 mb-4">
        ShikshaSetu is an AI-powered educational platform designed to make learning accessible
        in multiple Indian languages. Our platform simplifies complex educational content,
        translates it into 10 regional languages, and provides audio support for better comprehension.
      </p>
      <div className="grid md:grid-cols-2 gap-6 mt-8">
        <div className="glass-card p-6">
          <h3 className="text-xl font-semibold mb-3">✨ Features</h3>
          <ul className="space-y-2 text-gray-700">
            <li>• AI-powered text simplification</li>
            <li>• Multi-language translation</li>
            <li>• Text-to-speech audio generation</li>
            <li>• NCERT curriculum validation</li>
            <li>• Offline content access</li>
          </ul>
        </div>
        <div className="glass-card p-6">
          <h3 className="text-xl font-semibold mb-3">🎯 Supported</h3>
          <ul className="space-y-2 text-gray-700">
            <li>• Grades: 5-12</li>
            <li>• 6 subjects</li>
            <li>• 10 Indian languages</li>
            <li>• Real-time processing</li>
            <li>• Progress tracking</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AboutPage;
